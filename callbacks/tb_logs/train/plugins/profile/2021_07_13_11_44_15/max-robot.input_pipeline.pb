	??,&f3@??,&f3@!??,&f3@	?W??#???W??#??!?W??#??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:??,&f3@/??C???Aq;4,FA3@Yծ	i?A??rEagerKernelExecute 0*	\??????@2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?[??b?*@!??l???X@)?[??b?*@1??l???X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?l??Ԣ?!;??P????)?k%t?ę?1?o?+w??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchx?g?ɇ?!ω.2O???)x?g?ɇ?1ω.2O???:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap???а@*@!)&؞??X@)NA~6r?d?1????vϓ?:Preprocessing2F
Iterator::Model??d??!6??'a??)??%?2?b?1???m}??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?W??#??I??~??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	/??C???/??C???!/??C???      ??!       "      ??!       *      ??!       2	q;4,FA3@q;4,FA3@!q;4,FA3@:      ??!       B      ??!       J	ծ	i?A??ծ	i?A??!ծ	i?A??R      ??!       Z	ծ	i?A??ծ	i?A??!ծ	i?A??b      ??!       JCPU_ONLYY?W??#??b q??~??X@