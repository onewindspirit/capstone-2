	??k??9@??k??9@!??k??9@	?|?w????|?w???!?|?w???"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:??k??9@qU?wE???A???zT9@YE?[??b??rEagerKernelExecute 0*	u?6\?@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator??????$@!?6??X@)??????$@1?6??X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismN??1??!?l???)?Q*?	???1???m??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch8?q?曆?![??3???)8?q?曆?1[??3???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap?-??$@!e?X??X@)???ӹ?t?1???????:Preprocessing2F
Iterator::Model??ZC????!??c????)????V%q?1?ld?I???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?|?w???IB@ğ?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	qU?wE???qU?wE???!qU?wE???      ??!       "      ??!       *      ??!       2	???zT9@???zT9@!???zT9@:      ??!       B      ??!       J	E?[??b??E?[??b??!E?[??b??R      ??!       Z	E?[??b??E?[??b??!E?[??b??b      ??!       JCPU_ONLYY?|?w???b qB@ğ?X@