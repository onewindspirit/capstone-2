	U???B?4@U???B?4@!U???B?4@	????vQ??????vQ??!????vQ??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:U???B?4@?D?????Acc^G?4@Y;n??t˦?rEagerKernelExecute 0*	R??J?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorPs?"?'@!~????X@)Ps?"?'@1~????X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?
?.Ȗ?!?
*?????)^c?????1?? ?D??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??Z}u??!?vSz?~??)??Z}u??1?vSz?~??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapPō[?'@!???A??X@)?rK?!??1S??|???:Preprocessing2F
Iterator::Model???????!:??"|???) a??*f?1?}f??<??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9????vQ??I???DW?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?D??????D?????!?D?????      ??!       "      ??!       *      ??!       2	cc^G?4@cc^G?4@!cc^G?4@:      ??!       B      ??!       J	;n??t˦?;n??t˦?!;n??t˦?R      ??!       Z	;n??t˦?;n??t˦?!;n??t˦?b      ??!       JCPU_ONLYY????vQ??b q???DW?X@