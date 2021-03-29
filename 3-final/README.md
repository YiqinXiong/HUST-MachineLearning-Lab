## 题目6 语音性别识别

1. `voice_gender`文件夹下面为程序代码（直接复制的PyCharm工程文件夹）
2. `voice_gender/voice.csv`是数据集文件
3. `voice_gender/main.py`是程序源代码
4. 代码中的贝叶斯分类器定义为
    `def naive_bayes_classifier(row_id, is_log=True, is_weight=True)`
    其中`row_id`是数据集的行号，表示一个样本。`is_log`表示是否使用对数进行计算，`is_weight`表示是否引入属性权重进行优化。
    `is_log`和`is_weight`可以根据需要设置为`True`或`False`。