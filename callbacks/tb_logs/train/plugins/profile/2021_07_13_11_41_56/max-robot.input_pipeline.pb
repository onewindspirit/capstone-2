	=??@f/5@=??@f/5@!=??@f/5@	??M?A?????M?A???!??M?A???"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:=??@f/5@]?@?"??A?c?? 5@Y??)??F??rEagerKernelExecute 0*	?p=
??@@2]
&Iterator::Model::MaxIntraOpParallelism????~??!?-???	V@)?=~oӏ?1|U?gt G@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?2??(??!^FE@)?2??(??1^FE@:Preprocessing2F
Iterator::Model닄??K??!      Y@)????fdp?1??6Xʱ'@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??M?A???Ib?N?"?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	]?@?"??]?@?"??!]?@?"??      ??!       "      ??!       *      ??!       2	?c?? 5@?c?? 5@!?c?? 5@:      ??!       B      ??!       J	??)??F????)??F??!??)??F??R      ??!       Z	??)??F????)??F??!??)??F??b      ??!       JCPU_ONLYY??M?A???b qb?N?"?X@