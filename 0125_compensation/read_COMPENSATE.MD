现在这个文件夹里面有这些内容：

1ji_ccmba_preprocessor.py 和 2ji_ccmba_preprocessor.py 是用于处理训练数据的，motion_direct_train.py的训练需要ccmba处理完的数据。建议老师先用2ji_ccmba_preprocessor.py尝试一下。应该是指定好训练集路径和输出路径就OK了。这个处理完之后再去运行motion_direct_train.py。



gaussian_direct_train.py是用高斯噪声训练，有了genimage的sdv1.4普通训练集后可以直接使用的！使用这样的推荐指令运行：
"""
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr= 您的服务器IP \
    --master_port=8050 \
    gaussian_direct_train.py \
    --batch-size 32 \
    --output-dir ./output_distributed
"""

然后数据集和训练模糊比例等等参数在100-140行之间，其中BLUR_PROB = 0.2就是模糊数据集比例，建议训练三次不同参数0,0.1,0.2.（0即为干净）

motion_direct_train.py是用运动模糊训练的，使用前需要先用ccmba处理数据，然后指定好训练集路径和输出路径就OK了。使用这样的推荐指令运行：
"""
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr= 您的服务器IP \
    --master_port=8050 \
    motion_direct_train.py \
    --batch-size 32 \
    --output-dir ./output_distributed
"""

参数内容等等和gaussian完全一样的调整方式！

然后我这个多卡的代码设置需要老师运行后帮忙把关一下看看，双节点我不保证完全没有问题。