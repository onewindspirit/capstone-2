	??5?K?:@??5?K?:@!??5?K?:@	?dV^????dV^???!?dV^???"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:??5?K?:@v?1<???A?nK䂳:@Y?_???ܻ?rEagerKernelExecute 0*	@5^?	c?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator5_%??"@!?Ex?$?X@)5_%??"@1?Ex?$?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??:?? ??!p???X??)??:?? ??1p???X??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismv7Ou?ͨ?!??w??)^?c@?z??1>??IY-??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap????"@!q	zr??X@)G???R{??1?T6??:Preprocessing2F
Iterator::Model?Y5????!YG??Ɔ??)w?E?x?1???YEy??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?dV^???I????'?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	v?1<???v?1<???!v?1<???      ??!       "      ??!       *      ??!       2	?nK䂳:@?nK䂳:@!?nK䂳:@:      ??!       B      ??!       J	?_???ܻ??_???ܻ?!?_???ܻ?R      ??!       Z	?_???ܻ??_???ܻ?!?_???ܻ?b      ??!       JCPU_ONLYY?dV^???b q????'?X@