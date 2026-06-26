# 配置说明

当前 v3 的主要参数放在阶段化脚本环境变量中，便于直接改命令运行。后续如果某组参数稳定，可以在这里沉淀为 preset。

- 数据集事实来源：`dataset/classes.yaml`、`dataset/split.json`
- 单类别生产检测默认入口：`scripts/10_train_detector_single_prod.sh`
- 多类别预标注检测默认入口：`scripts/11_train_detector_multi_prelabel.sh`
- 分类离线裁剪默认入口：`scripts/02_dataset_crop_classifier_from_detector.ps1`

