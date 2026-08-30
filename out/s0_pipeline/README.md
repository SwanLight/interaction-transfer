# S0 —— 可视化管线

**S0 的产物是「能把画面录出来」这个能力本身**，不是某个文件。
所有场景录像在 `../s1_assets/`，策略录像在 `../s2_expert/`，
总报告是 `../report.html`。

`pipeline_ok.png` 是渲染管线跑通的一帧证据。

## 验证了什么

| 项 | 结果 |
|---|---|
| headless 渲染 | ✅ 需用 `AppLauncher(headless=True, enable_cameras=True)` |
| 录 mp4 | ✅ 6 个场景 |
| HTML 报告 | ✅ 自包含，视频内嵌 |
| WebRTC 串流 | ❌ **已取消**（D-28）：服务器与开发机不同网络 |

## 三条本环境特有的坑（`log/pitfalls.md` P-24）

1. 裸 `SimulationApp({"enable_cameras": True})` **无效**，必须用 `AppLauncher`
2. `sim.step(render=True)` 即使开了相机**仍会卡死**——物理走 `render=False`，
   出图时单独调 `sim.render()` + `cam.update()`
3. `cam.set_world_poses_from_view()` 卡死——相机位姿必须在
   `CameraCfg.OffsetCfg` 里静态给定
