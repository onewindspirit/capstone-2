	Q?f?6@Q?f?6@!Q?f?6@	???p??????p???!???p???"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:Q?f?6@?.?5#??A???W:?5@Y"T?????rEagerKernelExecute 0*	????>?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator
/???'@!??a?X@)
/???'@1??a?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch???V???!??^?G??)???V???1??^?G??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismR??/Ie??!??~?ո??)?vٯ;݉?1????)??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap????]?'@!?/????X@)??Tގpz?1;L?d?ī?:Preprocessing2F
Iterator::Model?c"?ٜ?!?A???L??)(?x?ߢc?1?s
ax???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9???p???I?G???X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?.?5#???.?5#??!?.?5#??      ??!       "      ??!       *      ??!       2	???W:?5@???W:?5@!???W:?5@:      ??!       B      ??!       J	"T?????"T?????!"T?????R      ??!       Z	"T?????"T?????!"T?????b      ??!       JCPU_ONLYY???p???b q?G???X@