Фн
џ■
B
AssignVariableOp
resource
value"dtype"
dtypetypeѕ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Џ
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(ѕ

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
Й
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ѕ
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.4.12unknown8ыб
|
conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv1/kernel
u
 conv1/kernel/Read/ReadVariableOpReadVariableOpconv1/kernel*&
_output_shapes
:*
dtype0
l

conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
conv1/bias
e
conv1/bias/Read/ReadVariableOpReadVariableOp
conv1/bias*
_output_shapes
:*
dtype0
|
conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2/kernel
u
 conv2/kernel/Read/ReadVariableOpReadVariableOpconv2/kernel*&
_output_shapes
:*
dtype0
l

conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
conv2/bias
e
conv2/bias/Read/ReadVariableOpReadVariableOp
conv2/bias*
_output_shapes
:*
dtype0
w
dense1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ *
shared_namedense1/kernel
p
!dense1/kernel/Read/ReadVariableOpReadVariableOpdense1/kernel*
_output_shapes
:	ђ *
dtype0
n
dense1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense1/bias
g
dense1/bias/Read/ReadVariableOpReadVariableOpdense1/bias*
_output_shapes
: *
dtype0

NoOpNoOp
ь
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*е
valueъBЏ Bћ
ќ
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4

signatures
#_self_saveable_object_factories
regularization_losses
		variables

trainable_variables
	keras_api
%
#_self_saveable_object_factories
Ї

kernel
bias
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
Ї

kernel
bias
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
w
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
Ї

 kernel
!bias
#"_self_saveable_object_factories
#regularization_losses
$	variables
%trainable_variables
&	keras_api
 
 
 
*
0
1
2
3
 4
!5
*
0
1
2
3
 4
!5
Г
'layer_metrics
regularization_losses
(layer_regularization_losses
)non_trainable_variables

*layers
		variables
+metrics

trainable_variables
 
XV
VARIABLE_VALUEconv1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
Г
,layer_metrics
-layer_regularization_losses
regularization_losses
.non_trainable_variables

/layers
	variables
0metrics
trainable_variables
XV
VARIABLE_VALUEconv2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
Г
1layer_metrics
2layer_regularization_losses
regularization_losses
3non_trainable_variables

4layers
	variables
5metrics
trainable_variables
 
 
 
 
Г
6layer_metrics
7layer_regularization_losses
regularization_losses
8non_trainable_variables

9layers
	variables
:metrics
trainable_variables
YW
VARIABLE_VALUEdense1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

 0
!1

 0
!1
Г
;layer_metrics
<layer_regularization_losses
#regularization_losses
=non_trainable_variables

>layers
$	variables
?metrics
%trainable_variables
 
 
 
#
0
1
2
3
4
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
ї
serving_default_input_imgPlaceholder*/
_output_shapes
:         @@*
dtype0*$
shape:         @@
ќ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_imgconv1/kernel
conv1/biasconv2/kernel
conv2/biasdense1/kerneldense1/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8ѓ *.
f)R'
%__inference_signature_wrapper_9298232
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
№
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename conv1/kernel/Read/ReadVariableOpconv1/bias/Read/ReadVariableOp conv2/kernel/Read/ReadVariableOpconv2/bias/Read/ReadVariableOp!dense1/kernel/Read/ReadVariableOpdense1/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *)
f$R"
 __inference__traced_save_9298414
Ы
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1/kernel
conv1/biasconv2/kernel
conv2/biasdense1/kerneldense1/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *,
f'R%
#__inference__traced_restore_9298442░э
═

█
B__inference_conv1_layer_call_and_return_conditional_losses_9298081

inputs#
conv2d_readvariableop_kernel_17"
biasadd_readvariableop_bias_17
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpќ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_kernel_17*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2
Conv2DІ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_17*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         2
ReluЪ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         @@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @@
 
_user_specified_nameinputs
Ж
Х
)__inference_model_3_layer_call_fn_9298297

inputs
	kernel_17
bias_17
	kernel_18
bias_18
	kernel_19
bias_19
identityѕбStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinputs	kernel_17bias_17	kernel_18bias_18	kernel_19bias_19*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_model_3_layer_call_and_return_conditional_losses_92981852
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         @@::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @@
 
_user_specified_nameinputs
╝
`
D__inference_flatten_layer_call_and_return_conditional_losses_9298122

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"     	  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         ђ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
з
╣
)__inference_model_3_layer_call_fn_9298219
	input_img
	kernel_17
bias_17
	kernel_18
bias_18
	kernel_19
bias_19
identityѕбStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCall	input_img	kernel_17bias_17	kernel_18bias_18	kernel_19bias_19*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_model_3_layer_call_and_return_conditional_losses_92982102
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         @@::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
/
_output_shapes
:         @@
#
_user_specified_name	input_img
═
х
%__inference_signature_wrapper_9298232
	input_img
	kernel_17
bias_17
	kernel_18
bias_18
	kernel_19
bias_19
identityѕбStatefulPartitionedCallі
StatefulPartitionedCallStatefulPartitionedCall	input_img	kernel_17bias_17	kernel_18bias_18	kernel_19bias_19*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8ѓ *+
f&R$
"__inference__wrapped_model_92980662
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         @@::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
/
_output_shapes
:         @@
#
_user_specified_name	input_img
Ф
┴
D__inference_model_3_layer_call_and_return_conditional_losses_9298168
	input_img
conv1_kernel_17
conv1_bias_17
conv2_kernel_18
conv2_bias_18
dense1_kernel_19
dense1_bias_19
identityѕбconv1/StatefulPartitionedCallбconv2/StatefulPartitionedCallбdense1/StatefulPartitionedCallџ
conv1/StatefulPartitionedCallStatefulPartitionedCall	input_imgconv1_kernel_17conv1_bias_17*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *K
fFRD
B__inference_conv1_layer_call_and_return_conditional_losses_92980812
conv1/StatefulPartitionedCallи
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0conv2_kernel_18conv2_bias_18*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *K
fFRD
B__inference_conv2_layer_call_and_return_conditional_losses_92981042
conv2/StatefulPartitionedCallЭ
flatten/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_92981222
flatten/PartitionedCall«
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_kernel_19dense1_bias_19*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *L
fGRE
C__inference_dense1_layer_call_and_return_conditional_losses_92981412 
dense1/StatefulPartitionedCall▄
IdentityIdentity'dense1/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^dense1/StatefulPartitionedCall*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         @@::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall:Z V
/
_output_shapes
:         @@
#
_user_specified_name	input_img
╝
`
D__inference_flatten_layer_call_and_return_conditional_losses_9298350

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"     	  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         ђ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs
Ж
Х
)__inference_model_3_layer_call_fn_9298308

inputs
	kernel_17
bias_17
	kernel_18
bias_18
	kernel_19
bias_19
identityѕбStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinputs	kernel_17bias_17	kernel_18bias_18	kernel_19bias_19*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_model_3_layer_call_and_return_conditional_losses_92982102
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         @@::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @@
 
_user_specified_nameinputs
з
╣
)__inference_model_3_layer_call_fn_9298194
	input_img
	kernel_17
bias_17
	kernel_18
bias_18
	kernel_19
bias_19
identityѕбStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCall	input_img	kernel_17bias_17	kernel_18bias_18	kernel_19bias_19*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_model_3_layer_call_and_return_conditional_losses_92981852
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         @@::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
/
_output_shapes
:         @@
#
_user_specified_name	input_img
б
Й
D__inference_model_3_layer_call_and_return_conditional_losses_9298185

inputs
conv1_kernel_17
conv1_bias_17
conv2_kernel_18
conv2_bias_18
dense1_kernel_19
dense1_bias_19
identityѕбconv1/StatefulPartitionedCallбconv2/StatefulPartitionedCallбdense1/StatefulPartitionedCallЌ
conv1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_kernel_17conv1_bias_17*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *K
fFRD
B__inference_conv1_layer_call_and_return_conditional_losses_92980812
conv1/StatefulPartitionedCallи
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0conv2_kernel_18conv2_bias_18*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *K
fFRD
B__inference_conv2_layer_call_and_return_conditional_losses_92981042
conv2/StatefulPartitionedCallЭ
flatten/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_92981222
flatten/PartitionedCall«
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_kernel_19dense1_bias_19*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *L
fGRE
C__inference_dense1_layer_call_and_return_conditional_losses_92981412 
dense1/StatefulPartitionedCall▄
IdentityIdentity'dense1/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^dense1/StatefulPartitionedCall*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         @@::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall:W S
/
_output_shapes
:         @@
 
_user_specified_nameinputs
р$
я
"__inference__wrapped_model_9298066
	input_img1
-model_3_conv1_conv2d_readvariableop_kernel_170
,model_3_conv1_biasadd_readvariableop_bias_171
-model_3_conv2_conv2d_readvariableop_kernel_180
,model_3_conv2_biasadd_readvariableop_bias_182
.model_3_dense1_matmul_readvariableop_kernel_191
-model_3_dense1_biasadd_readvariableop_bias_19
identityѕб$model_3/conv1/BiasAdd/ReadVariableOpб#model_3/conv1/Conv2D/ReadVariableOpб$model_3/conv2/BiasAdd/ReadVariableOpб#model_3/conv2/Conv2D/ReadVariableOpб%model_3/dense1/BiasAdd/ReadVariableOpб$model_3/dense1/MatMul/ReadVariableOp└
#model_3/conv1/Conv2D/ReadVariableOpReadVariableOp-model_3_conv1_conv2d_readvariableop_kernel_17*&
_output_shapes
:*
dtype02%
#model_3/conv1/Conv2D/ReadVariableOpЛ
model_3/conv1/Conv2DConv2D	input_img+model_3/conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2
model_3/conv1/Conv2Dх
$model_3/conv1/BiasAdd/ReadVariableOpReadVariableOp,model_3_conv1_biasadd_readvariableop_bias_17*
_output_shapes
:*
dtype02&
$model_3/conv1/BiasAdd/ReadVariableOp└
model_3/conv1/BiasAddBiasAddmodel_3/conv1/Conv2D:output:0,model_3/conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
model_3/conv1/BiasAddі
model_3/conv1/ReluRelumodel_3/conv1/BiasAdd:output:0*
T0*/
_output_shapes
:         2
model_3/conv1/Relu└
#model_3/conv2/Conv2D/ReadVariableOpReadVariableOp-model_3_conv2_conv2d_readvariableop_kernel_18*&
_output_shapes
:*
dtype02%
#model_3/conv2/Conv2D/ReadVariableOpУ
model_3/conv2/Conv2DConv2D model_3/conv1/Relu:activations:0+model_3/conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2
model_3/conv2/Conv2Dх
$model_3/conv2/BiasAdd/ReadVariableOpReadVariableOp,model_3_conv2_biasadd_readvariableop_bias_18*
_output_shapes
:*
dtype02&
$model_3/conv2/BiasAdd/ReadVariableOp└
model_3/conv2/BiasAddBiasAddmodel_3/conv2/Conv2D:output:0,model_3/conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
model_3/conv2/BiasAddі
model_3/conv2/ReluRelumodel_3/conv2/BiasAdd:output:0*
T0*/
_output_shapes
:         2
model_3/conv2/Relu
model_3/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"     	  2
model_3/flatten/Const▓
model_3/flatten/ReshapeReshape model_3/conv2/Relu:activations:0model_3/flatten/Const:output:0*
T0*(
_output_shapes
:         ђ2
model_3/flatten/Reshape╝
$model_3/dense1/MatMul/ReadVariableOpReadVariableOp.model_3_dense1_matmul_readvariableop_kernel_19*
_output_shapes
:	ђ *
dtype02&
$model_3/dense1/MatMul/ReadVariableOp║
model_3/dense1/MatMulMatMul model_3/flatten/Reshape:output:0,model_3/dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
model_3/dense1/MatMulИ
%model_3/dense1/BiasAdd/ReadVariableOpReadVariableOp-model_3_dense1_biasadd_readvariableop_bias_19*
_output_shapes
: *
dtype02'
%model_3/dense1/BiasAdd/ReadVariableOpй
model_3/dense1/BiasAddBiasAddmodel_3/dense1/MatMul:product:0-model_3/dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
model_3/dense1/BiasAddЁ
model_3/dense1/ReluRelumodel_3/dense1/BiasAdd:output:0*
T0*'
_output_shapes
:          2
model_3/dense1/Reluя
IdentityIdentity!model_3/dense1/Relu:activations:0%^model_3/conv1/BiasAdd/ReadVariableOp$^model_3/conv1/Conv2D/ReadVariableOp%^model_3/conv2/BiasAdd/ReadVariableOp$^model_3/conv2/Conv2D/ReadVariableOp&^model_3/dense1/BiasAdd/ReadVariableOp%^model_3/dense1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         @@::::::2L
$model_3/conv1/BiasAdd/ReadVariableOp$model_3/conv1/BiasAdd/ReadVariableOp2J
#model_3/conv1/Conv2D/ReadVariableOp#model_3/conv1/Conv2D/ReadVariableOp2L
$model_3/conv2/BiasAdd/ReadVariableOp$model_3/conv2/BiasAdd/ReadVariableOp2J
#model_3/conv2/Conv2D/ReadVariableOp#model_3/conv2/Conv2D/ReadVariableOp2N
%model_3/dense1/BiasAdd/ReadVariableOp%model_3/dense1/BiasAdd/ReadVariableOp2L
$model_3/dense1/MatMul/ReadVariableOp$model_3/dense1/MatMul/ReadVariableOp:Z V
/
_output_shapes
:         @@
#
_user_specified_name	input_img
§
|
'__inference_conv1_layer_call_fn_9298326

inputs
	kernel_17
bias_17
identityѕбStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputs	kernel_17bias_17*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *K
fFRD
B__inference_conv1_layer_call_and_return_conditional_losses_92980812
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         @@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @@
 
_user_specified_nameinputs
­	
▄
C__inference_dense1_layer_call_and_return_conditional_losses_9298366

inputs#
matmul_readvariableop_kernel_19"
biasadd_readvariableop_bias_19
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_kernel_19*
_output_shapes
:	ђ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
MatMulІ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_19*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:          2
ReluЌ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
б
Й
D__inference_model_3_layer_call_and_return_conditional_losses_9298210

inputs
conv1_kernel_17
conv1_bias_17
conv2_kernel_18
conv2_bias_18
dense1_kernel_19
dense1_bias_19
identityѕбconv1/StatefulPartitionedCallбconv2/StatefulPartitionedCallбdense1/StatefulPartitionedCallЌ
conv1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_kernel_17conv1_bias_17*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *K
fFRD
B__inference_conv1_layer_call_and_return_conditional_losses_92980812
conv1/StatefulPartitionedCallи
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0conv2_kernel_18conv2_bias_18*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *K
fFRD
B__inference_conv2_layer_call_and_return_conditional_losses_92981042
conv2/StatefulPartitionedCallЭ
flatten/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_92981222
flatten/PartitionedCall«
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_kernel_19dense1_bias_19*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *L
fGRE
C__inference_dense1_layer_call_and_return_conditional_losses_92981412 
dense1/StatefulPartitionedCall▄
IdentityIdentity'dense1/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^dense1/StatefulPartitionedCall*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         @@::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall:W S
/
_output_shapes
:         @@
 
_user_specified_nameinputs
р
}
(__inference_dense1_layer_call_fn_9298373

inputs
	kernel_19
bias_19
identityѕбStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputs	kernel_19bias_19*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *L
fGRE
C__inference_dense1_layer_call_and_return_conditional_losses_92981412
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Ш
Ю
#__inference__traced_restore_9298442
file_prefix!
assignvariableop_conv1_kernel!
assignvariableop_1_conv1_bias#
assignvariableop_2_conv2_kernel!
assignvariableop_3_conv2_bias$
 assignvariableop_4_dense1_kernel"
assignvariableop_5_dense1_bias

identity_7ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_2бAssignVariableOp_3бAssignVariableOp_4бAssignVariableOp_5ы
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*§
valueзB­B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesю
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
RestoreV2/shape_and_slices╬
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identityю
AssignVariableOpAssignVariableOpassignvariableop_conv1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1б
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2ц
AssignVariableOp_2AssignVariableOpassignvariableop_2_conv2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3б
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Ц
AssignVariableOp_4AssignVariableOp assignvariableop_4_dense1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Б
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpС

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_6о

Identity_7IdentityIdentity_6:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5*
T0*
_output_shapes
: 2

Identity_7"!

identity_7Identity_7:output:0*-
_input_shapes
: ::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_5:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ф
┴
D__inference_model_3_layer_call_and_return_conditional_losses_9298154
	input_img
conv1_kernel_17
conv1_bias_17
conv2_kernel_18
conv2_bias_18
dense1_kernel_19
dense1_bias_19
identityѕбconv1/StatefulPartitionedCallбconv2/StatefulPartitionedCallбdense1/StatefulPartitionedCallџ
conv1/StatefulPartitionedCallStatefulPartitionedCall	input_imgconv1_kernel_17conv1_bias_17*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *K
fFRD
B__inference_conv1_layer_call_and_return_conditional_losses_92980812
conv1/StatefulPartitionedCallи
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0conv2_kernel_18conv2_bias_18*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *K
fFRD
B__inference_conv2_layer_call_and_return_conditional_losses_92981042
conv2/StatefulPartitionedCallЭ
flatten/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_92981222
flatten/PartitionedCall«
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_kernel_19dense1_bias_19*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *L
fGRE
C__inference_dense1_layer_call_and_return_conditional_losses_92981412 
dense1/StatefulPartitionedCall▄
IdentityIdentity'dense1/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^dense1/StatefulPartitionedCall*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         @@::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall:Z V
/
_output_shapes
:         @@
#
_user_specified_name	input_img
┐
Ю
D__inference_model_3_layer_call_and_return_conditional_losses_9298286

inputs)
%conv1_conv2d_readvariableop_kernel_17(
$conv1_biasadd_readvariableop_bias_17)
%conv2_conv2d_readvariableop_kernel_18(
$conv2_biasadd_readvariableop_bias_18*
&dense1_matmul_readvariableop_kernel_19)
%dense1_biasadd_readvariableop_bias_19
identityѕбconv1/BiasAdd/ReadVariableOpбconv1/Conv2D/ReadVariableOpбconv2/BiasAdd/ReadVariableOpбconv2/Conv2D/ReadVariableOpбdense1/BiasAdd/ReadVariableOpбdense1/MatMul/ReadVariableOpе
conv1/Conv2D/ReadVariableOpReadVariableOp%conv1_conv2d_readvariableop_kernel_17*&
_output_shapes
:*
dtype02
conv1/Conv2D/ReadVariableOpХ
conv1/Conv2DConv2Dinputs#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2
conv1/Conv2DЮ
conv1/BiasAdd/ReadVariableOpReadVariableOp$conv1_biasadd_readvariableop_bias_17*
_output_shapes
:*
dtype02
conv1/BiasAdd/ReadVariableOpа
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
conv1/BiasAddr

conv1/ReluReluconv1/BiasAdd:output:0*
T0*/
_output_shapes
:         2

conv1/Reluе
conv2/Conv2D/ReadVariableOpReadVariableOp%conv2_conv2d_readvariableop_kernel_18*&
_output_shapes
:*
dtype02
conv2/Conv2D/ReadVariableOp╚
conv2/Conv2DConv2Dconv1/Relu:activations:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2
conv2/Conv2DЮ
conv2/BiasAdd/ReadVariableOpReadVariableOp$conv2_biasadd_readvariableop_bias_18*
_output_shapes
:*
dtype02
conv2/BiasAdd/ReadVariableOpа
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
conv2/BiasAddr

conv2/ReluReluconv2/BiasAdd:output:0*
T0*/
_output_shapes
:         2

conv2/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"     	  2
flatten/Constњ
flatten/ReshapeReshapeconv2/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:         ђ2
flatten/Reshapeц
dense1/MatMul/ReadVariableOpReadVariableOp&dense1_matmul_readvariableop_kernel_19*
_output_shapes
:	ђ *
dtype02
dense1/MatMul/ReadVariableOpџ
dense1/MatMulMatMulflatten/Reshape:output:0$dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense1/MatMulа
dense1/BiasAdd/ReadVariableOpReadVariableOp%dense1_biasadd_readvariableop_bias_19*
_output_shapes
: *
dtype02
dense1/BiasAdd/ReadVariableOpЮ
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense1/BiasAddm
dense1/ReluReludense1/BiasAdd:output:0*
T0*'
_output_shapes
:          2
dense1/Reluд
IdentityIdentitydense1/Relu:activations:0^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp^dense1/BiasAdd/ReadVariableOp^dense1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         @@::::::2<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp2:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp2<
conv2/BiasAdd/ReadVariableOpconv2/BiasAdd/ReadVariableOp2:
conv2/Conv2D/ReadVariableOpconv2/Conv2D/ReadVariableOp2>
dense1/BiasAdd/ReadVariableOpdense1/BiasAdd/ReadVariableOp2<
dense1/MatMul/ReadVariableOpdense1/MatMul/ReadVariableOp:W S
/
_output_shapes
:         @@
 
_user_specified_nameinputs
┐
Ю
D__inference_model_3_layer_call_and_return_conditional_losses_9298259

inputs)
%conv1_conv2d_readvariableop_kernel_17(
$conv1_biasadd_readvariableop_bias_17)
%conv2_conv2d_readvariableop_kernel_18(
$conv2_biasadd_readvariableop_bias_18*
&dense1_matmul_readvariableop_kernel_19)
%dense1_biasadd_readvariableop_bias_19
identityѕбconv1/BiasAdd/ReadVariableOpбconv1/Conv2D/ReadVariableOpбconv2/BiasAdd/ReadVariableOpбconv2/Conv2D/ReadVariableOpбdense1/BiasAdd/ReadVariableOpбdense1/MatMul/ReadVariableOpе
conv1/Conv2D/ReadVariableOpReadVariableOp%conv1_conv2d_readvariableop_kernel_17*&
_output_shapes
:*
dtype02
conv1/Conv2D/ReadVariableOpХ
conv1/Conv2DConv2Dinputs#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2
conv1/Conv2DЮ
conv1/BiasAdd/ReadVariableOpReadVariableOp$conv1_biasadd_readvariableop_bias_17*
_output_shapes
:*
dtype02
conv1/BiasAdd/ReadVariableOpа
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
conv1/BiasAddr

conv1/ReluReluconv1/BiasAdd:output:0*
T0*/
_output_shapes
:         2

conv1/Reluе
conv2/Conv2D/ReadVariableOpReadVariableOp%conv2_conv2d_readvariableop_kernel_18*&
_output_shapes
:*
dtype02
conv2/Conv2D/ReadVariableOp╚
conv2/Conv2DConv2Dconv1/Relu:activations:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2
conv2/Conv2DЮ
conv2/BiasAdd/ReadVariableOpReadVariableOp$conv2_biasadd_readvariableop_bias_18*
_output_shapes
:*
dtype02
conv2/BiasAdd/ReadVariableOpа
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2
conv2/BiasAddr

conv2/ReluReluconv2/BiasAdd:output:0*
T0*/
_output_shapes
:         2

conv2/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"     	  2
flatten/Constњ
flatten/ReshapeReshapeconv2/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:         ђ2
flatten/Reshapeц
dense1/MatMul/ReadVariableOpReadVariableOp&dense1_matmul_readvariableop_kernel_19*
_output_shapes
:	ђ *
dtype02
dense1/MatMul/ReadVariableOpџ
dense1/MatMulMatMulflatten/Reshape:output:0$dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense1/MatMulа
dense1/BiasAdd/ReadVariableOpReadVariableOp%dense1_biasadd_readvariableop_bias_19*
_output_shapes
: *
dtype02
dense1/BiasAdd/ReadVariableOpЮ
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense1/BiasAddm
dense1/ReluReludense1/BiasAdd:output:0*
T0*'
_output_shapes
:          2
dense1/Reluд
IdentityIdentitydense1/Relu:activations:0^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp^dense1/BiasAdd/ReadVariableOp^dense1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:         @@::::::2<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp2:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp2<
conv2/BiasAdd/ReadVariableOpconv2/BiasAdd/ReadVariableOp2:
conv2/Conv2D/ReadVariableOpconv2/Conv2D/ReadVariableOp2>
dense1/BiasAdd/ReadVariableOpdense1/BiasAdd/ReadVariableOp2<
dense1/MatMul/ReadVariableOpdense1/MatMul/ReadVariableOp:W S
/
_output_shapes
:         @@
 
_user_specified_nameinputs
═

█
B__inference_conv1_layer_call_and_return_conditional_losses_9298319

inputs#
conv2d_readvariableop_kernel_17"
biasadd_readvariableop_bias_17
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpќ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_kernel_17*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2
Conv2DІ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_17*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         2
ReluЪ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         @@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @@
 
_user_specified_nameinputs
§
|
'__inference_conv2_layer_call_fn_9298344

inputs
	kernel_18
bias_18
identityѕбStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputs	kernel_18bias_18*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8ѓ *K
fFRD
B__inference_conv2_layer_call_and_return_conditional_losses_92981042
StatefulPartitionedCallќ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
═

█
B__inference_conv2_layer_call_and_return_conditional_losses_9298104

inputs#
conv2d_readvariableop_kernel_18"
biasadd_readvariableop_bias_18
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpќ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_kernel_18*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2
Conv2DІ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_18*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         2
ReluЪ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
═

█
B__inference_conv2_layer_call_and_return_conditional_losses_9298337

inputs#
conv2d_readvariableop_kernel_18"
biasadd_readvariableop_bias_18
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOpќ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_kernel_18*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpц
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         *
paddingVALID*
strides
2
Conv2DІ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_18*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpѕ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:         2
ReluЪ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:         2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:         ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         
 
_user_specified_nameinputs
­	
▄
C__inference_dense1_layer_call_and_return_conditional_losses_9298141

inputs#
matmul_readvariableop_kernel_19"
biasadd_readvariableop_bias_19
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_kernel_19*
_output_shapes
:	ђ *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
MatMulІ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_19*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:          2
ReluЌ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ђ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
▄
э
 __inference__traced_save_9298414
file_prefix+
'savev2_conv1_kernel_read_readvariableop)
%savev2_conv1_bias_read_readvariableop+
'savev2_conv2_kernel_read_readvariableop)
%savev2_conv2_bias_read_readvariableop,
(savev2_dense1_kernel_read_readvariableop*
&savev2_dense1_bias_read_readvariableop
savev2_const

identity_1ѕбMergeV2CheckpointsЈ
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1І
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameв
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*§
valueзB­B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesќ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
SaveV2/shape_and_slices▓
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_conv1_kernel_read_readvariableop%savev2_conv1_bias_read_readvariableop'savev2_conv2_kernel_read_readvariableop%savev2_conv2_bias_read_readvariableop(savev2_dense1_kernel_read_readvariableop&savev2_dense1_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
	22
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesА
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*X
_input_shapesG
E: :::::	ђ : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	ђ : 

_output_shapes
: :

_output_shapes
: 
Е
E
)__inference_flatten_layer_call_fn_9298355

inputs
identity╚
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8ѓ *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_92981222
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :W S
/
_output_shapes
:         
 
_user_specified_nameinputs"▒L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*х
serving_defaultА
G
	input_img:
serving_default_input_img:0         @@:
dense10
StatefulPartitionedCall:0          tensorflow/serving/predict:«ъ
╩.
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4

signatures
#_self_saveable_object_factories
regularization_losses
		variables

trainable_variables
	keras_api
@__call__
*A&call_and_return_all_conditional_losses
B_default_save_signature"┌+
_tf_keras_networkЙ+{"class_name": "Functional", "name": "model_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_img"}, "name": "input_img", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [8, 8]}, "strides": {"class_name": "__tuple__", "items": [4, 4]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1", "inbound_nodes": [[["input_img", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2", "inbound_nodes": [[["conv1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["conv2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense1", "inbound_nodes": [[["flatten", 0, 0, {}]]]}], "input_layers": [["input_img", 0, 0]], "output_layers": [["dense1", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_img"}, "name": "input_img", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [8, 8]}, "strides": {"class_name": "__tuple__", "items": [4, 4]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1", "inbound_nodes": [[["input_img", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2", "inbound_nodes": [[["conv1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["conv2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense1", "inbound_nodes": [[["flatten", 0, 0, {}]]]}], "input_layers": [["input_img", 0, 0]], "output_layers": [["dense1", 0, 0]]}}}
б
#_self_saveable_object_factories"Щ
_tf_keras_input_layer┌{"class_name": "InputLayer", "name": "input_img", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_img"}}
Ј


kernel
bias
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
C__call__
*D&call_and_return_all_conditional_losses"┼
_tf_keras_layerФ{"class_name": "Conv2D", "name": "conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [8, 8]}, "strides": {"class_name": "__tuple__", "items": [4, 4]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 3]}}
љ


kernel
bias
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
E__call__
*F&call_and_return_all_conditional_losses"к
_tf_keras_layerг{"class_name": "Conv2D", "name": "conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 15, 15, 8]}}
Є
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
G__call__
*H&call_and_return_all_conditional_losses"М
_tf_keras_layer╣{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
Ќ

 kernel
!bias
#"_self_saveable_object_factories
#regularization_losses
$	variables
%trainable_variables
&	keras_api
I__call__
*J&call_and_return_all_conditional_losses"═
_tf_keras_layer│{"class_name": "Dense", "name": "dense1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2304}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2304]}}
,
Kserving_default"
signature_map
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
J
0
1
2
3
 4
!5"
trackable_list_wrapper
J
0
1
2
3
 4
!5"
trackable_list_wrapper
╩
'layer_metrics
regularization_losses
(layer_regularization_losses
)non_trainable_variables

*layers
		variables
+metrics

trainable_variables
@__call__
B_default_save_signature
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
&:$2conv1/kernel
:2
conv1/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
Г
,layer_metrics
-layer_regularization_losses
regularization_losses
.non_trainable_variables

/layers
	variables
0metrics
trainable_variables
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
&:$2conv2/kernel
:2
conv2/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
Г
1layer_metrics
2layer_regularization_losses
regularization_losses
3non_trainable_variables

4layers
	variables
5metrics
trainable_variables
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Г
6layer_metrics
7layer_regularization_losses
regularization_losses
8non_trainable_variables

9layers
	variables
:metrics
trainable_variables
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
 :	ђ 2dense1/kernel
: 2dense1/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
Г
;layer_metrics
<layer_regularization_losses
#regularization_losses
=non_trainable_variables

>layers
$	variables
?metrics
%trainable_variables
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ы2№
)__inference_model_3_layer_call_fn_9298308
)__inference_model_3_layer_call_fn_9298194
)__inference_model_3_layer_call_fn_9298219
)__inference_model_3_layer_call_fn_9298297└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
я2█
D__inference_model_3_layer_call_and_return_conditional_losses_9298168
D__inference_model_3_layer_call_and_return_conditional_losses_9298259
D__inference_model_3_layer_call_and_return_conditional_losses_9298154
D__inference_model_3_layer_call_and_return_conditional_losses_9298286└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ж2у
"__inference__wrapped_model_9298066└
І▓Є
FullArgSpec
argsџ 
varargsjargs
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *0б-
+і(
	input_img         @@
Л2╬
'__inference_conv1_layer_call_fn_9298326б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
В2ж
B__inference_conv1_layer_call_and_return_conditional_losses_9298319б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Л2╬
'__inference_conv2_layer_call_fn_9298344б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
В2ж
B__inference_conv2_layer_call_and_return_conditional_losses_9298337б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
М2л
)__inference_flatten_layer_call_fn_9298355б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_flatten_layer_call_and_return_conditional_losses_9298350б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
м2¤
(__inference_dense1_layer_call_fn_9298373б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ь2Ж
C__inference_dense1_layer_call_and_return_conditional_losses_9298366б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
╬B╦
%__inference_signature_wrapper_9298232	input_img"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 Џ
"__inference__wrapped_model_9298066u !:б7
0б-
+і(
	input_img         @@
ф "/ф,
*
dense1 і
dense1          ▓
B__inference_conv1_layer_call_and_return_conditional_losses_9298319l7б4
-б*
(і%
inputs         @@
ф "-б*
#і 
0         
џ і
'__inference_conv1_layer_call_fn_9298326_7б4
-б*
(і%
inputs         @@
ф " і         ▓
B__inference_conv2_layer_call_and_return_conditional_losses_9298337l7б4
-б*
(і%
inputs         
ф "-б*
#і 
0         
џ і
'__inference_conv2_layer_call_fn_9298344_7б4
-б*
(і%
inputs         
ф " і         ц
C__inference_dense1_layer_call_and_return_conditional_losses_9298366] !0б-
&б#
!і
inputs         ђ
ф "%б"
і
0          
џ |
(__inference_dense1_layer_call_fn_9298373P !0б-
&б#
!і
inputs         ђ
ф "і          Е
D__inference_flatten_layer_call_and_return_conditional_losses_9298350a7б4
-б*
(і%
inputs         
ф "&б#
і
0         ђ
џ Ђ
)__inference_flatten_layer_call_fn_9298355T7б4
-б*
(і%
inputs         
ф "і         ђ╗
D__inference_model_3_layer_call_and_return_conditional_losses_9298154s !Bб?
8б5
+і(
	input_img         @@
p

 
ф "%б"
і
0          
џ ╗
D__inference_model_3_layer_call_and_return_conditional_losses_9298168s !Bб?
8б5
+і(
	input_img         @@
p 

 
ф "%б"
і
0          
џ И
D__inference_model_3_layer_call_and_return_conditional_losses_9298259p !?б<
5б2
(і%
inputs         @@
p

 
ф "%б"
і
0          
џ И
D__inference_model_3_layer_call_and_return_conditional_losses_9298286p !?б<
5б2
(і%
inputs         @@
p 

 
ф "%б"
і
0          
џ Њ
)__inference_model_3_layer_call_fn_9298194f !Bб?
8б5
+і(
	input_img         @@
p

 
ф "і          Њ
)__inference_model_3_layer_call_fn_9298219f !Bб?
8б5
+і(
	input_img         @@
p 

 
ф "і          љ
)__inference_model_3_layer_call_fn_9298297c !?б<
5б2
(і%
inputs         @@
p

 
ф "і          љ
)__inference_model_3_layer_call_fn_9298308c !?б<
5б2
(і%
inputs         @@
p 

 
ф "і          г
%__inference_signature_wrapper_9298232ѓ !GбD
б 
=ф:
8
	input_img+і(
	input_img         @@"/ф,
*
dense1 і
dense1          