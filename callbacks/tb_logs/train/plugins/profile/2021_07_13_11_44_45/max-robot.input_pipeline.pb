	 ???.3@ ???.3@! ???.3@	'?P????'?P????!'?P????"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails: ???.3@????M??A?)??z3@Y?Tkaک?rEagerKernelExecute 0**??.??@)      ?=2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator???d??+@!????P?X@)???d??+@1????P?X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?_?5?!??!?Ͻ???)-[닄???1???5???:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::PrefetchZd;?O???!?<?ER??)Zd;?O???1?<?ER??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapy?0DN?+@!v?Z???X@)S=??Mj?1%????ϗ?:Preprocessing2F
Iterator::Model1'h??'??!]7J?d??)i;???.h?1t??c4???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9'?P????I?~?h'?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????M??????M??!????M??      ??!       "      ??!       *      ??!       2	?)??z3@?)??z3@!?)??z3@:      ??!       B      ??!       J	?Tkaک??Tkaک?!?Tkaک?R      ??!       Z	?Tkaک??Tkaک?!?Tkaک?b      ??!       JCPU_ONLYY'?P????b q?~?h'?X@