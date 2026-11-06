# COMP0245: Behaviour Cloning for Mobile-Manipulator Reaching

This project trains separate base and arm MLPs from expert demonstrations and evaluates reaching performance in PyBullet. A hybrid variant adds bounded terminal feedback correction.

All project code is located in `final_project/robotnik_arm_task/`.

See [the project README](final_project/robotnik_arm_task/README.md) for environment setup, data collection, training, evaluation, generated outputs and code attribution.

Pretrained weights and the required `metadata.json` and `episodes.json` are included (`final_project/robotnik_arm_task/data/expert_*/`). Full training trajectories are not included. After environment setup, evaluate directly without data collection or training:

```bash
python final_project/robotnik_arm_task/final_evaluation.py
```
