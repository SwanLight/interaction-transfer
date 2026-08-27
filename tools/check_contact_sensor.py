"""R-4 v3: filter 目标改为 kinematic 刚体地板"""
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import torch
out = open("/tmp/verify3.txt","w")
def P(*a): print(*a, file=out, flush=True)

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass

MASS, MU, G = 0.5, 0.4, 9.81
MAT = sim_utils.RigidBodyMaterialCfg(static_friction=MU, dynamic_friction=MU, restitution=0.0)

@configclass
class SceneCfg(InteractiveSceneCfg):
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=1000.0))
    # 关键改动：地板是 kinematic 刚体，不是静态碰撞体
    floor = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Floor",
        spawn=sim_utils.CuboidCfg(size=(4.0,4.0,0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1000.0),
            collision_props=sim_utils.CollisionPropertiesCfg(), physics_material=MAT,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3,0.3,0.3)),
            activate_contact_sensors=True),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0,0.0,-0.1)))
    cube = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(size=(0.1,0.1,0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(), physics_material=MAT,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8,0.2,0.2)),
            activate_contact_sensors=True),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0,0.0,0.06)))
    contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Cube",
        track_pose=True, track_contact_points=True,
        max_contact_data_count_per_prim=16,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Floor"],
        update_period=0.0, history_length=0)

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/120, device="cuda:0"))
scene = InteractiveScene(SceneCfg(num_envs=2, env_spacing=6.0)); sim.reset()
cube: RigidObject = scene["cube"]; cs: ContactSensor = scene["contact"]
dt = sim.get_physics_dt()
def steps(n, f=None):
    for _ in range(n):
        if f is not None: cube.set_external_force_and_torque(f, torch.zeros_like(f))
        cube.write_data_to_sim(); scene.write_data_to_sim(); sim.step(); scene.update(dt)

P("="*70); P("R-4 v3   filter 目标 = kinematic 刚体地板"); P("="*70)
steps(300)
cd = cs.contact_physx_view.get_contact_data(dt=dt)
P(f"接触对数量 counts = {cd[4].flatten().tolist()}   <-- v2 里是 [0,0]")
P(f"force_matrix_w = {[round(x,4) for x in cs.data.force_matrix_w[0,0,0].tolist()]}")
P(f"contact_pos_w NaN 比例 = {torch.isnan(cs.data.contact_pos_w).float().mean():.3f}")
if not torch.isnan(cs.data.contact_pos_w[0,0,0]).any():
    P(f"contact_pos_w[0,0,0] = {[round(x,5) for x in cs.data.contact_pos_w[0,0,0].tolist()]}")

P(f"\n### stick/slide 判定（理论阈值 mu*m*g = {MU*MASS*G:.3f} N）###")
P(f"{'F(N)':>6} {'|v|(m/s)':>10} {'Fn(N)':>8} {'|Ff|(N)':>9} {'接触对':>7}  模式")
for F in (1.0, 1.5, 2.5, 3.0):
    scene.reset(); steps(200)
    push = torch.zeros(scene.num_envs,1,3, device=sim.device); push[:,0,0] = F
    steps(60, push)
    v = cube.data.root_lin_vel_w[0]; net = cs.data.net_forces_w[0,0]
    ff = cs.contact_physx_view.get_friction_data(dt=dt)[0].view(-1,3)
    n = cs.contact_physx_view.get_contact_data(dt=dt)[4].flatten()[0].item()
    tot = ff.sum(dim=0)
    P(f"{F:6.1f} {v.norm():10.4f} {net[2]:8.3f} {tot.norm():9.4f} {n:7d}  {'SLIDE' if v[:2].norm()>0.05 else 'STICK'}")
    P(f"       摩擦力向量=[{tot[0]:+.4f},{tot[1]:+.4f},{tot[2]:+.4f}]")

P("\n### 接触点 / 法向（Interaction Region 需要）###")
scene.reset(); steps(200)
cd = cs.contact_physx_view.get_contact_data(dt=dt)
nm = ["forces","points","normals","separations","counts","start_idx"]
for i,x in enumerate(cd):
    if x.dtype.is_floating_point:
        v2 = x.view(-1, x.shape[-1])
        P(f"  {nm[i]:11s} 非零行={int((v2.norm(dim=-1)>1e-9).sum()):3d}/{len(v2):3d}  首个非零={[round(y,5) for y in v2[(v2.norm(dim=-1)>1e-9).nonzero()[0,0]].tolist()] if (v2.norm(dim=-1)>1e-9).any() else 'N/A'}")
    else:
        P(f"  {nm[i]:11s} = {x.flatten().tolist()}")
P("="*70)
out.close(); app.close()
