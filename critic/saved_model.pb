??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
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
delete_old_dirsbool(?
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
dtypetype?
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
?
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
executor_typestring ?
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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12unknown8??
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
shape:	? *
shared_namedense1/kernel
p
!dense1/kernel/Read/ReadVariableOpReadVariableOpdense1/kernel*
_output_shapes
:	? *
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
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:#*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:#*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

:*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?!
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?!
value?!B?  B? 
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8


signatures
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
%
#_self_saveable_object_factories
?

kernel
bias
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
?

kernel
bias
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
w
#_self_saveable_object_factories
 regularization_losses
!	variables
"trainable_variables
#	keras_api
?

$kernel
%bias
#&_self_saveable_object_factories
'regularization_losses
(	variables
)trainable_variables
*	keras_api
%
#+_self_saveable_object_factories
w
#,_self_saveable_object_factories
-regularization_losses
.	variables
/trainable_variables
0	keras_api
?

1kernel
2bias
#3_self_saveable_object_factories
4regularization_losses
5	variables
6trainable_variables
7	keras_api
?

8kernel
9bias
#:_self_saveable_object_factories
;regularization_losses
<	variables
=trainable_variables
>	keras_api
 
 
 
F
0
1
2
3
$4
%5
16
27
88
99
F
0
1
2
3
$4
%5
16
27
88
99
?
?layer_metrics
regularization_losses
@layer_regularization_losses
Anon_trainable_variables

Blayers
	variables
Cmetrics
trainable_variables
 
XV
VARIABLE_VALUEconv1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
?
Dlayer_metrics
Elayer_regularization_losses
regularization_losses
Fnon_trainable_variables

Glayers
	variables
Hmetrics
trainable_variables
XV
VARIABLE_VALUEconv2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
?
Ilayer_metrics
Jlayer_regularization_losses
regularization_losses
Knon_trainable_variables

Llayers
	variables
Mmetrics
trainable_variables
 
 
 
 
?
Nlayer_metrics
Olayer_regularization_losses
 regularization_losses
Pnon_trainable_variables

Qlayers
!	variables
Rmetrics
"trainable_variables
YW
VARIABLE_VALUEdense1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

$0
%1

$0
%1
?
Slayer_metrics
Tlayer_regularization_losses
'regularization_losses
Unon_trainable_variables

Vlayers
(	variables
Wmetrics
)trainable_variables
 
 
 
 
 
?
Xlayer_metrics
Ylayer_regularization_losses
-regularization_losses
Znon_trainable_variables

[layers
.	variables
\metrics
/trainable_variables
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

10
21

10
21
?
]layer_metrics
^layer_regularization_losses
4regularization_losses
_non_trainable_variables

`layers
5	variables
ametrics
6trainable_variables
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

80
91

80
91
?
blayer_metrics
clayer_regularization_losses
;regularization_losses
dnon_trainable_variables

elayers
<	variables
fmetrics
=trainable_variables
 
 
 
?
0
1
2
3
4
5
6
7
	8
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
|
serving_default_input_actPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
serving_default_input_imgPlaceholder*/
_output_shapes
:?????????@@*
dtype0*$
shape:?????????@@
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_actserving_default_input_imgconv1/kernel
conv1/biasconv2/kernel
conv2/biasdense1/kerneldense1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8? *.
f)R'
%__inference_signature_wrapper_9296691
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename conv1/kernel/Read/ReadVariableOpconv1/bias/Read/ReadVariableOp conv2/kernel/Read/ReadVariableOpconv2/bias/Read/ReadVariableOp!dense1/kernel/Read/ReadVariableOpdense1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOpConst*
Tin
2*
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
GPU2 *0J 8? *)
f$R"
 __inference__traced_save_9296976
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1/kernel
conv1/biasconv2/kernel
conv2/biasdense1/kerneldense1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias*
Tin
2*
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
GPU2 *0J 8? *,
f'R%
#__inference__traced_restore_9297016??
?
t
J__inference_concatenate_2_layer_call_and_return_conditional_losses_9296519

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????#2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????#2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:????????? :?????????:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
B__inference_conv2_layer_call_and_return_conditional_losses_9296838

inputs#
conv2d_readvariableop_kernel_13"
biasadd_readvariableop_bias_13
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_kernel_13*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_13*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
D__inference_dense_3_layer_call_and_return_conditional_losses_9296561

inputs#
matmul_readvariableop_kernel_16"
biasadd_readvariableop_bias_16
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_kernel_16*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_16*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
~
)__inference_dense_2_layer_call_fn_9296905

inputs
	kernel_15
bias_15
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs	kernel_15bias_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_92965392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????#::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????#
 
_user_specified_nameinputs
?"
?
D__inference_model_2_layer_call_and_return_conditional_losses_9296660

inputs
inputs_1
conv1_kernel_12
conv1_bias_12
conv2_kernel_13
conv2_bias_13
dense1_kernel_14
dense1_bias_14
dense_2_kernel_15
dense_2_bias_15
dense_3_kernel_16
dense_3_bias_16
identity??conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?dense1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_kernel_12conv1_bias_12*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_conv1_layer_call_and_return_conditional_losses_92964402
conv1/StatefulPartitionedCall?
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0conv2_kernel_13conv2_bias_13*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_conv2_layer_call_and_return_conditional_losses_92964632
conv2/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_92964812
flatten/PartitionedCall?
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_kernel_14dense1_bias_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_dense1_layer_call_and_return_conditional_losses_92965002 
dense1/StatefulPartitionedCall?
concatenate_2/PartitionedCallPartitionedCall'dense1/StatefulPartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????#* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_concatenate_2_layer_call_and_return_conditional_losses_92965192
concatenate_2/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_2_kernel_15dense_2_bias_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_92965392!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_kernel_16dense_3_bias_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_92965612!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^dense1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:?????????@@:?????????::::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?<
?
"__inference__wrapped_model_9296424
	input_img
	input_act1
-model_2_conv1_conv2d_readvariableop_kernel_120
,model_2_conv1_biasadd_readvariableop_bias_121
-model_2_conv2_conv2d_readvariableop_kernel_130
,model_2_conv2_biasadd_readvariableop_bias_132
.model_2_dense1_matmul_readvariableop_kernel_141
-model_2_dense1_biasadd_readvariableop_bias_143
/model_2_dense_2_matmul_readvariableop_kernel_152
.model_2_dense_2_biasadd_readvariableop_bias_153
/model_2_dense_3_matmul_readvariableop_kernel_162
.model_2_dense_3_biasadd_readvariableop_bias_16
identity??$model_2/conv1/BiasAdd/ReadVariableOp?#model_2/conv1/Conv2D/ReadVariableOp?$model_2/conv2/BiasAdd/ReadVariableOp?#model_2/conv2/Conv2D/ReadVariableOp?%model_2/dense1/BiasAdd/ReadVariableOp?$model_2/dense1/MatMul/ReadVariableOp?&model_2/dense_2/BiasAdd/ReadVariableOp?%model_2/dense_2/MatMul/ReadVariableOp?&model_2/dense_3/BiasAdd/ReadVariableOp?%model_2/dense_3/MatMul/ReadVariableOp?
#model_2/conv1/Conv2D/ReadVariableOpReadVariableOp-model_2_conv1_conv2d_readvariableop_kernel_12*&
_output_shapes
:*
dtype02%
#model_2/conv1/Conv2D/ReadVariableOp?
model_2/conv1/Conv2DConv2D	input_img+model_2/conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
model_2/conv1/Conv2D?
$model_2/conv1/BiasAdd/ReadVariableOpReadVariableOp,model_2_conv1_biasadd_readvariableop_bias_12*
_output_shapes
:*
dtype02&
$model_2/conv1/BiasAdd/ReadVariableOp?
model_2/conv1/BiasAddBiasAddmodel_2/conv1/Conv2D:output:0,model_2/conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
model_2/conv1/BiasAdd?
model_2/conv1/ReluRelumodel_2/conv1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
model_2/conv1/Relu?
#model_2/conv2/Conv2D/ReadVariableOpReadVariableOp-model_2_conv2_conv2d_readvariableop_kernel_13*&
_output_shapes
:*
dtype02%
#model_2/conv2/Conv2D/ReadVariableOp?
model_2/conv2/Conv2DConv2D model_2/conv1/Relu:activations:0+model_2/conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
model_2/conv2/Conv2D?
$model_2/conv2/BiasAdd/ReadVariableOpReadVariableOp,model_2_conv2_biasadd_readvariableop_bias_13*
_output_shapes
:*
dtype02&
$model_2/conv2/BiasAdd/ReadVariableOp?
model_2/conv2/BiasAddBiasAddmodel_2/conv2/Conv2D:output:0,model_2/conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
model_2/conv2/BiasAdd?
model_2/conv2/ReluRelumodel_2/conv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
model_2/conv2/Relu
model_2/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  2
model_2/flatten/Const?
model_2/flatten/ReshapeReshape model_2/conv2/Relu:activations:0model_2/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
model_2/flatten/Reshape?
$model_2/dense1/MatMul/ReadVariableOpReadVariableOp.model_2_dense1_matmul_readvariableop_kernel_14*
_output_shapes
:	? *
dtype02&
$model_2/dense1/MatMul/ReadVariableOp?
model_2/dense1/MatMulMatMul model_2/flatten/Reshape:output:0,model_2/dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
model_2/dense1/MatMul?
%model_2/dense1/BiasAdd/ReadVariableOpReadVariableOp-model_2_dense1_biasadd_readvariableop_bias_14*
_output_shapes
: *
dtype02'
%model_2/dense1/BiasAdd/ReadVariableOp?
model_2/dense1/BiasAddBiasAddmodel_2/dense1/MatMul:product:0-model_2/dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
model_2/dense1/BiasAdd?
model_2/dense1/ReluRelumodel_2/dense1/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
model_2/dense1/Relu?
!model_2/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_2/concatenate_2/concat/axis?
model_2/concatenate_2/concatConcatV2!model_2/dense1/Relu:activations:0	input_act*model_2/concatenate_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????#2
model_2/concatenate_2/concat?
%model_2/dense_2/MatMul/ReadVariableOpReadVariableOp/model_2_dense_2_matmul_readvariableop_kernel_15*
_output_shapes

:#*
dtype02'
%model_2/dense_2/MatMul/ReadVariableOp?
model_2/dense_2/MatMulMatMul%model_2/concatenate_2/concat:output:0-model_2/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_2/dense_2/MatMul?
&model_2/dense_2/BiasAdd/ReadVariableOpReadVariableOp.model_2_dense_2_biasadd_readvariableop_bias_15*
_output_shapes
:*
dtype02(
&model_2/dense_2/BiasAdd/ReadVariableOp?
model_2/dense_2/BiasAddBiasAdd model_2/dense_2/MatMul:product:0.model_2/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_2/dense_2/BiasAdd?
model_2/dense_2/ReluRelu model_2/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_2/dense_2/Relu?
%model_2/dense_3/MatMul/ReadVariableOpReadVariableOp/model_2_dense_3_matmul_readvariableop_kernel_16*
_output_shapes

:*
dtype02'
%model_2/dense_3/MatMul/ReadVariableOp?
model_2/dense_3/MatMulMatMul"model_2/dense_2/Relu:activations:0-model_2/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_2/dense_3/MatMul?
&model_2/dense_3/BiasAdd/ReadVariableOpReadVariableOp.model_2_dense_3_biasadd_readvariableop_bias_16*
_output_shapes
:*
dtype02(
&model_2/dense_3/BiasAdd/ReadVariableOp?
model_2/dense_3/BiasAddBiasAdd model_2/dense_3/MatMul:product:0.model_2/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_2/dense_3/BiasAdd?
IdentityIdentity model_2/dense_3/BiasAdd:output:0%^model_2/conv1/BiasAdd/ReadVariableOp$^model_2/conv1/Conv2D/ReadVariableOp%^model_2/conv2/BiasAdd/ReadVariableOp$^model_2/conv2/Conv2D/ReadVariableOp&^model_2/dense1/BiasAdd/ReadVariableOp%^model_2/dense1/MatMul/ReadVariableOp'^model_2/dense_2/BiasAdd/ReadVariableOp&^model_2/dense_2/MatMul/ReadVariableOp'^model_2/dense_3/BiasAdd/ReadVariableOp&^model_2/dense_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:?????????@@:?????????::::::::::2L
$model_2/conv1/BiasAdd/ReadVariableOp$model_2/conv1/BiasAdd/ReadVariableOp2J
#model_2/conv1/Conv2D/ReadVariableOp#model_2/conv1/Conv2D/ReadVariableOp2L
$model_2/conv2/BiasAdd/ReadVariableOp$model_2/conv2/BiasAdd/ReadVariableOp2J
#model_2/conv2/Conv2D/ReadVariableOp#model_2/conv2/Conv2D/ReadVariableOp2N
%model_2/dense1/BiasAdd/ReadVariableOp%model_2/dense1/BiasAdd/ReadVariableOp2L
$model_2/dense1/MatMul/ReadVariableOp$model_2/dense1/MatMul/ReadVariableOp2P
&model_2/dense_2/BiasAdd/ReadVariableOp&model_2/dense_2/BiasAdd/ReadVariableOp2N
%model_2/dense_2/MatMul/ReadVariableOp%model_2/dense_2/MatMul/ReadVariableOp2P
&model_2/dense_3/BiasAdd/ReadVariableOp&model_2/dense_3/BiasAdd/ReadVariableOp2N
%model_2/dense_3/MatMul/ReadVariableOp%model_2/dense_3/MatMul/ReadVariableOp:Z V
/
_output_shapes
:?????????@@
#
_user_specified_name	input_img:RN
'
_output_shapes
:?????????
#
_user_specified_name	input_act
?

?
B__inference_conv1_layer_call_and_return_conditional_losses_9296440

inputs#
conv2d_readvariableop_kernel_12"
biasadd_readvariableop_bias_12
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_kernel_12*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_12*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
|
'__inference_conv2_layer_call_fn_9296845

inputs
	kernel_13
bias_13
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs	kernel_13bias_13*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_conv2_layer_call_and_return_conditional_losses_92964632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
D__inference_dense_2_layer_call_and_return_conditional_losses_9296539

inputs#
matmul_readvariableop_kernel_15"
biasadd_readvariableop_bias_15
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_kernel_15*
_output_shapes

:#*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_15*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????#::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????#
 
_user_specified_nameinputs
?"
?
D__inference_model_2_layer_call_and_return_conditional_losses_9296574
	input_img
	input_act
conv1_kernel_12
conv1_bias_12
conv2_kernel_13
conv2_bias_13
dense1_kernel_14
dense1_bias_14
dense_2_kernel_15
dense_2_bias_15
dense_3_kernel_16
dense_3_bias_16
identity??conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?dense1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCall	input_imgconv1_kernel_12conv1_bias_12*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_conv1_layer_call_and_return_conditional_losses_92964402
conv1/StatefulPartitionedCall?
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0conv2_kernel_13conv2_bias_13*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_conv2_layer_call_and_return_conditional_losses_92964632
conv2/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_92964812
flatten/PartitionedCall?
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_kernel_14dense1_bias_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_dense1_layer_call_and_return_conditional_losses_92965002 
dense1/StatefulPartitionedCall?
concatenate_2/PartitionedCallPartitionedCall'dense1/StatefulPartitionedCall:output:0	input_act*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????#* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_concatenate_2_layer_call_and_return_conditional_losses_92965192
concatenate_2/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_2_kernel_15dense_2_bias_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_92965392!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_kernel_16dense_3_bias_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_92965612!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^dense1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:?????????@@:?????????::::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:Z V
/
_output_shapes
:?????????@@
#
_user_specified_name	input_img:RN
'
_output_shapes
:?????????
#
_user_specified_name	input_act
?
`
D__inference_flatten_layer_call_and_return_conditional_losses_9296851

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?"
?
D__inference_model_2_layer_call_and_return_conditional_losses_9296596
	input_img
	input_act
conv1_kernel_12
conv1_bias_12
conv2_kernel_13
conv2_bias_13
dense1_kernel_14
dense1_bias_14
dense_2_kernel_15
dense_2_bias_15
dense_3_kernel_16
dense_3_bias_16
identity??conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?dense1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCall	input_imgconv1_kernel_12conv1_bias_12*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_conv1_layer_call_and_return_conditional_losses_92964402
conv1/StatefulPartitionedCall?
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0conv2_kernel_13conv2_bias_13*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_conv2_layer_call_and_return_conditional_losses_92964632
conv2/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_92964812
flatten/PartitionedCall?
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_kernel_14dense1_bias_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_dense1_layer_call_and_return_conditional_losses_92965002 
dense1/StatefulPartitionedCall?
concatenate_2/PartitionedCallPartitionedCall'dense1/StatefulPartitionedCall:output:0	input_act*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????#* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_concatenate_2_layer_call_and_return_conditional_losses_92965192
concatenate_2/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_2_kernel_15dense_2_bias_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_92965392!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_kernel_16dense_3_bias_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_92965612!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^dense1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:?????????@@:?????????::::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:Z V
/
_output_shapes
:?????????@@
#
_user_specified_name	input_img:RN
'
_output_shapes
:?????????
#
_user_specified_name	input_act
?	
?
)__inference_model_2_layer_call_fn_9296673
	input_img
	input_act
	kernel_12
bias_12
	kernel_13
bias_13
	kernel_14
bias_14
	kernel_15
bias_15
	kernel_16
bias_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	input_img	input_act	kernel_12bias_12	kernel_13bias_13	kernel_14bias_14	kernel_15bias_15	kernel_16bias_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_model_2_layer_call_and_return_conditional_losses_92966602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:?????????@@:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
/
_output_shapes
:?????????@@
#
_user_specified_name	input_img:RN
'
_output_shapes
:?????????
#
_user_specified_name	input_act
?

?
B__inference_conv2_layer_call_and_return_conditional_losses_9296463

inputs#
conv2d_readvariableop_kernel_13"
biasadd_readvariableop_bias_13
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_kernel_13*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_13*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?"
?
D__inference_model_2_layer_call_and_return_conditional_losses_9296622

inputs
inputs_1
conv1_kernel_12
conv1_bias_12
conv2_kernel_13
conv2_bias_13
dense1_kernel_14
dense1_bias_14
dense_2_kernel_15
dense_2_bias_15
dense_3_kernel_16
dense_3_bias_16
identity??conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?dense1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_kernel_12conv1_bias_12*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_conv1_layer_call_and_return_conditional_losses_92964402
conv1/StatefulPartitionedCall?
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0conv2_kernel_13conv2_bias_13*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_conv2_layer_call_and_return_conditional_losses_92964632
conv2/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall&conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_92964812
flatten/PartitionedCall?
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_kernel_14dense1_bias_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_dense1_layer_call_and_return_conditional_losses_92965002 
dense1/StatefulPartitionedCall?
concatenate_2/PartitionedCallPartitionedCall'dense1/StatefulPartitionedCall:output:0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????#* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_concatenate_2_layer_call_and_return_conditional_losses_92965192
concatenate_2/PartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_2_kernel_15dense_2_bias_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_92965392!
dense_2/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_kernel_16dense_3_bias_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_92965612!
dense_3/StatefulPartitionedCall?
IdentityIdentity(dense_3/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall^dense1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:?????????@@:?????????::::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
E
)__inference_flatten_layer_call_fn_9296856

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_92964812
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
|
'__inference_conv1_layer_call_fn_9296827

inputs
	kernel_12
bias_12
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs	kernel_12bias_12*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *K
fFRD
B__inference_conv1_layer_call_and_return_conditional_losses_92964402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?	
?
D__inference_dense_2_layer_call_and_return_conditional_losses_9296898

inputs#
matmul_readvariableop_kernel_15"
biasadd_readvariableop_bias_15
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_kernel_15*
_output_shapes

:#*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_15*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????#::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????#
 
_user_specified_nameinputs
?!
?
 __inference__traced_save_9296976
file_prefix+
'savev2_conv1_kernel_read_readvariableop)
%savev2_conv1_bias_read_readvariableop+
'savev2_conv2_kernel_read_readvariableop)
%savev2_conv2_bias_read_readvariableop,
(savev2_dense1_kernel_read_readvariableop*
&savev2_dense1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
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
Const_1?
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
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_conv1_kernel_read_readvariableop%savev2_conv1_bias_read_readvariableop'savev2_conv2_kernel_read_readvariableop%savev2_conv2_bias_read_readvariableop(savev2_dense1_kernel_read_readvariableop&savev2_dense1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
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

identity_1Identity_1:output:0*x
_input_shapesg
e: :::::	? : :#:::: 2(
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
:	? : 

_output_shapes
: :$ 

_output_shapes

:#: 

_output_shapes
::$	 

_output_shapes

:: 


_output_shapes
::

_output_shapes
: 
?
`
D__inference_flatten_layer_call_and_return_conditional_losses_9296481

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
D__inference_dense_3_layer_call_and_return_conditional_losses_9296915

inputs#
matmul_readvariableop_kernel_16"
biasadd_readvariableop_bias_16
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_kernel_16*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_16*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?-
?
#__inference__traced_restore_9297016
file_prefix!
assignvariableop_conv1_kernel!
assignvariableop_1_conv1_bias#
assignvariableop_2_conv2_kernel!
assignvariableop_3_conv2_bias$
 assignvariableop_4_dense1_kernel"
assignvariableop_5_dense1_bias%
!assignvariableop_6_dense_2_kernel#
assignvariableop_7_dense_2_bias%
!assignvariableop_8_dense_3_kernel#
assignvariableop_9_dense_3_bias
identity_11??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*@
_output_shapes.
,:::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_conv1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_conv2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp assignvariableop_4_dense1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_3_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_3_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_10?
Identity_11IdentityIdentity_10:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_11"#
identity_11Identity_11:output:0*=
_input_shapes,
*: ::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
~
)__inference_dense_3_layer_call_fn_9296922

inputs
	kernel_16
bias_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs	kernel_16bias_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_92965612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
)__inference_model_2_layer_call_fn_9296809
inputs_0
inputs_1
	kernel_12
bias_12
	kernel_13
bias_13
	kernel_14
bias_14
	kernel_15
bias_15
	kernel_16
bias_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1	kernel_12bias_12	kernel_13bias_13	kernel_14bias_14	kernel_15bias_15	kernel_16bias_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_model_2_layer_call_and_return_conditional_losses_92966602
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:?????????@@:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????@@
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?3
?
D__inference_model_2_layer_call_and_return_conditional_losses_9296777
inputs_0
inputs_1)
%conv1_conv2d_readvariableop_kernel_12(
$conv1_biasadd_readvariableop_bias_12)
%conv2_conv2d_readvariableop_kernel_13(
$conv2_biasadd_readvariableop_bias_13*
&dense1_matmul_readvariableop_kernel_14)
%dense1_biasadd_readvariableop_bias_14+
'dense_2_matmul_readvariableop_kernel_15*
&dense_2_biasadd_readvariableop_bias_15+
'dense_3_matmul_readvariableop_kernel_16*
&dense_3_biasadd_readvariableop_bias_16
identity??conv1/BiasAdd/ReadVariableOp?conv1/Conv2D/ReadVariableOp?conv2/BiasAdd/ReadVariableOp?conv2/Conv2D/ReadVariableOp?dense1/BiasAdd/ReadVariableOp?dense1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?
conv1/Conv2D/ReadVariableOpReadVariableOp%conv1_conv2d_readvariableop_kernel_12*&
_output_shapes
:*
dtype02
conv1/Conv2D/ReadVariableOp?
conv1/Conv2DConv2Dinputs_0#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1/Conv2D?
conv1/BiasAdd/ReadVariableOpReadVariableOp$conv1_biasadd_readvariableop_bias_12*
_output_shapes
:*
dtype02
conv1/BiasAdd/ReadVariableOp?
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv1/BiasAddr

conv1/ReluReluconv1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2

conv1/Relu?
conv2/Conv2D/ReadVariableOpReadVariableOp%conv2_conv2d_readvariableop_kernel_13*&
_output_shapes
:*
dtype02
conv2/Conv2D/ReadVariableOp?
conv2/Conv2DConv2Dconv1/Relu:activations:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv2/Conv2D?
conv2/BiasAdd/ReadVariableOpReadVariableOp$conv2_biasadd_readvariableop_bias_13*
_output_shapes
:*
dtype02
conv2/BiasAdd/ReadVariableOp?
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2/BiasAddr

conv2/ReluReluconv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2

conv2/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  2
flatten/Const?
flatten/ReshapeReshapeconv2/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
dense1/MatMul/ReadVariableOpReadVariableOp&dense1_matmul_readvariableop_kernel_14*
_output_shapes
:	? *
dtype02
dense1/MatMul/ReadVariableOp?
dense1/MatMulMatMulflatten/Reshape:output:0$dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense1/MatMul?
dense1/BiasAdd/ReadVariableOpReadVariableOp%dense1_biasadd_readvariableop_bias_14*
_output_shapes
: *
dtype02
dense1/BiasAdd/ReadVariableOp?
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense1/BiasAddm
dense1/ReluReludense1/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense1/Relux
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_2/concat/axis?
concatenate_2/concatConcatV2dense1/Relu:activations:0inputs_1"concatenate_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????#2
concatenate_2/concat?
dense_2/MatMul/ReadVariableOpReadVariableOp'dense_2_matmul_readvariableop_kernel_15*
_output_shapes

:#*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMulconcatenate_2/concat:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp&dense_2_biasadd_readvariableop_bias_15*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_2/Relu?
dense_3/MatMul/ReadVariableOpReadVariableOp'dense_3_matmul_readvariableop_kernel_16*
_output_shapes

:*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp&dense_3_biasadd_readvariableop_bias_16*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/BiasAdd?
IdentityIdentitydense_3/BiasAdd:output:0^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp^dense1/BiasAdd/ReadVariableOp^dense1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:?????????@@:?????????::::::::::2<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp2:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp2<
conv2/BiasAdd/ReadVariableOpconv2/BiasAdd/ReadVariableOp2:
conv2/Conv2D/ReadVariableOpconv2/Conv2D/ReadVariableOp2>
dense1/BiasAdd/ReadVariableOpdense1/BiasAdd/ReadVariableOp2<
dense1/MatMul/ReadVariableOpdense1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:Y U
/
_output_shapes
:?????????@@
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?

?
B__inference_conv1_layer_call_and_return_conditional_losses_9296820

inputs#
conv2d_readvariableop_kernel_12"
biasadd_readvariableop_bias_12
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_kernel_12*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_12*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
[
/__inference_concatenate_2_layer_call_fn_9296887
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????#* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_concatenate_2_layer_call_and_return_conditional_losses_92965192
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????#2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:????????? :?????????:Q M
'
_output_shapes
:????????? 
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
}
(__inference_dense1_layer_call_fn_9296874

inputs
	kernel_14
bias_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs	kernel_14bias_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:????????? *$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_dense1_layer_call_and_return_conditional_losses_92965002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
C__inference_dense1_layer_call_and_return_conditional_losses_9296867

inputs#
matmul_readvariableop_kernel_14"
biasadd_readvariableop_bias_14
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_kernel_14*
_output_shapes
:	? *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_14*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
)__inference_model_2_layer_call_fn_9296635
	input_img
	input_act
	kernel_12
bias_12
	kernel_13
bias_13
	kernel_14
bias_14
	kernel_15
bias_15
	kernel_16
bias_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	input_img	input_act	kernel_12bias_12	kernel_13bias_13	kernel_14bias_14	kernel_15bias_15	kernel_16bias_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_model_2_layer_call_and_return_conditional_losses_92966222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:?????????@@:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
/
_output_shapes
:?????????@@
#
_user_specified_name	input_img:RN
'
_output_shapes
:?????????
#
_user_specified_name	input_act
?	
?
%__inference_signature_wrapper_9296691
	input_act
	input_img
	kernel_12
bias_12
	kernel_13
bias_13
	kernel_14
bias_14
	kernel_15
bias_15
	kernel_16
bias_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	input_img	input_act	kernel_12bias_12	kernel_13bias_13	kernel_14bias_14	kernel_15bias_15	kernel_16bias_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8? *+
f&R$
"__inference__wrapped_model_92964242
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:?????????:?????????@@::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:?????????
#
_user_specified_name	input_act:ZV
/
_output_shapes
:?????????@@
#
_user_specified_name	input_img
?
v
J__inference_concatenate_2_layer_call_and_return_conditional_losses_9296881
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????#2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????#2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:????????? :?????????:Q M
'
_output_shapes
:????????? 
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?	
?
C__inference_dense1_layer_call_and_return_conditional_losses_9296500

inputs#
matmul_readvariableop_kernel_14"
biasadd_readvariableop_bias_14
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_kernel_14*
_output_shapes
:	? *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_14*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
)__inference_model_2_layer_call_fn_9296793
inputs_0
inputs_1
	kernel_12
bias_12
	kernel_13
bias_13
	kernel_14
bias_14
	kernel_15
bias_15
	kernel_16
bias_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1	kernel_12bias_12	kernel_13bias_13	kernel_14bias_14	kernel_15bias_15	kernel_16bias_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_model_2_layer_call_and_return_conditional_losses_92966222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:?????????@@:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:?????????@@
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?3
?
D__inference_model_2_layer_call_and_return_conditional_losses_9296734
inputs_0
inputs_1)
%conv1_conv2d_readvariableop_kernel_12(
$conv1_biasadd_readvariableop_bias_12)
%conv2_conv2d_readvariableop_kernel_13(
$conv2_biasadd_readvariableop_bias_13*
&dense1_matmul_readvariableop_kernel_14)
%dense1_biasadd_readvariableop_bias_14+
'dense_2_matmul_readvariableop_kernel_15*
&dense_2_biasadd_readvariableop_bias_15+
'dense_3_matmul_readvariableop_kernel_16*
&dense_3_biasadd_readvariableop_bias_16
identity??conv1/BiasAdd/ReadVariableOp?conv1/Conv2D/ReadVariableOp?conv2/BiasAdd/ReadVariableOp?conv2/Conv2D/ReadVariableOp?dense1/BiasAdd/ReadVariableOp?dense1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?
conv1/Conv2D/ReadVariableOpReadVariableOp%conv1_conv2d_readvariableop_kernel_12*&
_output_shapes
:*
dtype02
conv1/Conv2D/ReadVariableOp?
conv1/Conv2DConv2Dinputs_0#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1/Conv2D?
conv1/BiasAdd/ReadVariableOpReadVariableOp$conv1_biasadd_readvariableop_bias_12*
_output_shapes
:*
dtype02
conv1/BiasAdd/ReadVariableOp?
conv1/BiasAddBiasAddconv1/Conv2D:output:0$conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv1/BiasAddr

conv1/ReluReluconv1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2

conv1/Relu?
conv2/Conv2D/ReadVariableOpReadVariableOp%conv2_conv2d_readvariableop_kernel_13*&
_output_shapes
:*
dtype02
conv2/Conv2D/ReadVariableOp?
conv2/Conv2DConv2Dconv1/Relu:activations:0#conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv2/Conv2D?
conv2/BiasAdd/ReadVariableOpReadVariableOp$conv2_biasadd_readvariableop_bias_13*
_output_shapes
:*
dtype02
conv2/BiasAdd/ReadVariableOp?
conv2/BiasAddBiasAddconv2/Conv2D:output:0$conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2/BiasAddr

conv2/ReluReluconv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2

conv2/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  2
flatten/Const?
flatten/ReshapeReshapeconv2/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
dense1/MatMul/ReadVariableOpReadVariableOp&dense1_matmul_readvariableop_kernel_14*
_output_shapes
:	? *
dtype02
dense1/MatMul/ReadVariableOp?
dense1/MatMulMatMulflatten/Reshape:output:0$dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense1/MatMul?
dense1/BiasAdd/ReadVariableOpReadVariableOp%dense1_biasadd_readvariableop_bias_14*
_output_shapes
: *
dtype02
dense1/BiasAdd/ReadVariableOp?
dense1/BiasAddBiasAdddense1/MatMul:product:0%dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense1/BiasAddm
dense1/ReluReludense1/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
dense1/Relux
concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_2/concat/axis?
concatenate_2/concatConcatV2dense1/Relu:activations:0inputs_1"concatenate_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????#2
concatenate_2/concat?
dense_2/MatMul/ReadVariableOpReadVariableOp'dense_2_matmul_readvariableop_kernel_15*
_output_shapes

:#*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMulconcatenate_2/concat:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp&dense_2_biasadd_readvariableop_bias_15*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_2/Relu?
dense_3/MatMul/ReadVariableOpReadVariableOp'dense_3_matmul_readvariableop_kernel_16*
_output_shapes

:*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp&dense_3_biasadd_readvariableop_bias_16*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/BiasAdd?
IdentityIdentitydense_3/BiasAdd:output:0^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp^dense1/BiasAdd/ReadVariableOp^dense1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*i
_input_shapesX
V:?????????@@:?????????::::::::::2<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp2:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp2<
conv2/BiasAdd/ReadVariableOpconv2/BiasAdd/ReadVariableOp2:
conv2/Conv2D/ReadVariableOpconv2/Conv2D/ReadVariableOp2>
dense1/BiasAdd/ReadVariableOpdense1/BiasAdd/ReadVariableOp2<
dense1/MatMul/ReadVariableOpdense1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:Y U
/
_output_shapes
:?????????@@
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
?
	input_act2
serving_default_input_act:0?????????
G
	input_img:
serving_default_input_img:0?????????@@;
dense_30
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?G
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8


signatures
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
g__call__
*h&call_and_return_all_conditional_losses
i_default_save_signature"?D
_tf_keras_network?C{"class_name": "Functional", "name": "model_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_img"}, "name": "input_img", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [8, 8]}, "strides": {"class_name": "__tuple__", "items": [4, 4]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1", "inbound_nodes": [[["input_img", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2", "inbound_nodes": [[["conv1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["conv2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense1", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_act"}, "name": "input_act", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate_2", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_2", "inbound_nodes": [[["dense1", 0, 0, {}], ["input_act", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["concatenate_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}], "input_layers": [["input_img", 0, 0], ["input_act", 0, 0]], "output_layers": [["dense_3", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 3]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 64, 64, 3]}, {"class_name": "TensorShape", "items": [null, 3]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_img"}, "name": "input_img", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [8, 8]}, "strides": {"class_name": "__tuple__", "items": [4, 4]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1", "inbound_nodes": [[["input_img", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2", "inbound_nodes": [[["conv1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["conv2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense1", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_act"}, "name": "input_act", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate_2", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_2", "inbound_nodes": [[["dense1", 0, 0, {}], ["input_act", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["concatenate_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}], "input_layers": [["input_img", 0, 0], ["input_act", 0, 0]], "output_layers": [["dense_3", 0, 0]]}}}
?
#_self_saveable_object_factories"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_img", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_img"}}
?


kernel
bias
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
j__call__
*k&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [8, 8]}, "strides": {"class_name": "__tuple__", "items": [4, 4]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 3]}}
?


kernel
bias
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
l__call__
*m&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 15, 15, 8]}}
?
#_self_saveable_object_factories
 regularization_losses
!	variables
"trainable_variables
#	keras_api
n__call__
*o&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

$kernel
%bias
#&_self_saveable_object_factories
'regularization_losses
(	variables
)trainable_variables
*	keras_api
p__call__
*q&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2304}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2304]}}
?
#+_self_saveable_object_factories"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_act", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_act"}}
?
#,_self_saveable_object_factories
-regularization_losses
.	variables
/trainable_variables
0	keras_api
r__call__
*s&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_2", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 32]}, {"class_name": "TensorShape", "items": [null, 3]}]}
?

1kernel
2bias
#3_self_saveable_object_factories
4regularization_losses
5	variables
6trainable_variables
7	keras_api
t__call__
*u&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 35}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35]}}
?

8kernel
9bias
#:_self_saveable_object_factories
;regularization_losses
<	variables
=trainable_variables
>	keras_api
v__call__
*w&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20]}}
,
xserving_default"
signature_map
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
f
0
1
2
3
$4
%5
16
27
88
99"
trackable_list_wrapper
f
0
1
2
3
$4
%5
16
27
88
99"
trackable_list_wrapper
?
?layer_metrics
regularization_losses
@layer_regularization_losses
Anon_trainable_variables

Blayers
	variables
Cmetrics
trainable_variables
g__call__
i_default_save_signature
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
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
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Dlayer_metrics
Elayer_regularization_losses
regularization_losses
Fnon_trainable_variables

Glayers
	variables
Hmetrics
trainable_variables
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
&:$2conv2/kernel
:2
conv2/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Ilayer_metrics
Jlayer_regularization_losses
regularization_losses
Knon_trainable_variables

Llayers
	variables
Mmetrics
trainable_variables
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Nlayer_metrics
Olayer_regularization_losses
 regularization_losses
Pnon_trainable_variables

Qlayers
!	variables
Rmetrics
"trainable_variables
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
 :	? 2dense1/kernel
: 2dense1/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
?
Slayer_metrics
Tlayer_regularization_losses
'regularization_losses
Unon_trainable_variables

Vlayers
(	variables
Wmetrics
)trainable_variables
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Xlayer_metrics
Ylayer_regularization_losses
-regularization_losses
Znon_trainable_variables

[layers
.	variables
\metrics
/trainable_variables
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
 :#2dense_2/kernel
:2dense_2/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
?
]layer_metrics
^layer_regularization_losses
4regularization_losses
_non_trainable_variables

`layers
5	variables
ametrics
6trainable_variables
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
 :2dense_3/kernel
:2dense_3/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
?
blayer_metrics
clayer_regularization_losses
;regularization_losses
dnon_trainable_variables

elayers
<	variables
fmetrics
=trainable_variables
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
	8"
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
?2?
)__inference_model_2_layer_call_fn_9296673
)__inference_model_2_layer_call_fn_9296809
)__inference_model_2_layer_call_fn_9296793
)__inference_model_2_layer_call_fn_9296635?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_model_2_layer_call_and_return_conditional_losses_9296734
D__inference_model_2_layer_call_and_return_conditional_losses_9296574
D__inference_model_2_layer_call_and_return_conditional_losses_9296777
D__inference_model_2_layer_call_and_return_conditional_losses_9296596?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
"__inference__wrapped_model_9296424?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *Z?W
U?R
+?(
	input_img?????????@@
#? 
	input_act?????????
?2?
'__inference_conv1_layer_call_fn_9296827?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_conv1_layer_call_and_return_conditional_losses_9296820?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_conv2_layer_call_fn_9296845?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_conv2_layer_call_and_return_conditional_losses_9296838?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_flatten_layer_call_fn_9296856?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_flatten_layer_call_and_return_conditional_losses_9296851?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense1_layer_call_fn_9296874?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense1_layer_call_and_return_conditional_losses_9296867?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_concatenate_2_layer_call_fn_9296887?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_concatenate_2_layer_call_and_return_conditional_losses_9296881?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_2_layer_call_fn_9296905?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_2_layer_call_and_return_conditional_losses_9296898?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_3_layer_call_fn_9296922?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_3_layer_call_and_return_conditional_losses_9296915?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
%__inference_signature_wrapper_9296691	input_act	input_img"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
"__inference__wrapped_model_9296424?
$%1289d?a
Z?W
U?R
+?(
	input_img?????????@@
#? 
	input_act?????????
? "1?.
,
dense_3!?
dense_3??????????
J__inference_concatenate_2_layer_call_and_return_conditional_losses_9296881?Z?W
P?M
K?H
"?
inputs/0????????? 
"?
inputs/1?????????
? "%?"
?
0?????????#
? ?
/__inference_concatenate_2_layer_call_fn_9296887vZ?W
P?M
K?H
"?
inputs/0????????? 
"?
inputs/1?????????
? "??????????#?
B__inference_conv1_layer_call_and_return_conditional_losses_9296820l7?4
-?*
(?%
inputs?????????@@
? "-?*
#? 
0?????????
? ?
'__inference_conv1_layer_call_fn_9296827_7?4
-?*
(?%
inputs?????????@@
? " ???????????
B__inference_conv2_layer_call_and_return_conditional_losses_9296838l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
'__inference_conv2_layer_call_fn_9296845_7?4
-?*
(?%
inputs?????????
? " ???????????
C__inference_dense1_layer_call_and_return_conditional_losses_9296867]$%0?-
&?#
!?
inputs??????????
? "%?"
?
0????????? 
? |
(__inference_dense1_layer_call_fn_9296874P$%0?-
&?#
!?
inputs??????????
? "?????????? ?
D__inference_dense_2_layer_call_and_return_conditional_losses_9296898\12/?,
%?"
 ?
inputs?????????#
? "%?"
?
0?????????
? |
)__inference_dense_2_layer_call_fn_9296905O12/?,
%?"
 ?
inputs?????????#
? "???????????
D__inference_dense_3_layer_call_and_return_conditional_losses_9296915\89/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? |
)__inference_dense_3_layer_call_fn_9296922O89/?,
%?"
 ?
inputs?????????
? "???????????
D__inference_flatten_layer_call_and_return_conditional_losses_9296851a7?4
-?*
(?%
inputs?????????
? "&?#
?
0??????????
? ?
)__inference_flatten_layer_call_fn_9296856T7?4
-?*
(?%
inputs?????????
? "????????????
D__inference_model_2_layer_call_and_return_conditional_losses_9296574?
$%1289l?i
b?_
U?R
+?(
	input_img?????????@@
#? 
	input_act?????????
p

 
? "%?"
?
0?????????
? ?
D__inference_model_2_layer_call_and_return_conditional_losses_9296596?
$%1289l?i
b?_
U?R
+?(
	input_img?????????@@
#? 
	input_act?????????
p 

 
? "%?"
?
0?????????
? ?
D__inference_model_2_layer_call_and_return_conditional_losses_9296734?
$%1289j?g
`?]
S?P
*?'
inputs/0?????????@@
"?
inputs/1?????????
p

 
? "%?"
?
0?????????
? ?
D__inference_model_2_layer_call_and_return_conditional_losses_9296777?
$%1289j?g
`?]
S?P
*?'
inputs/0?????????@@
"?
inputs/1?????????
p 

 
? "%?"
?
0?????????
? ?
)__inference_model_2_layer_call_fn_9296635?
$%1289l?i
b?_
U?R
+?(
	input_img?????????@@
#? 
	input_act?????????
p

 
? "???????????
)__inference_model_2_layer_call_fn_9296673?
$%1289l?i
b?_
U?R
+?(
	input_img?????????@@
#? 
	input_act?????????
p 

 
? "???????????
)__inference_model_2_layer_call_fn_9296793?
$%1289j?g
`?]
S?P
*?'
inputs/0?????????@@
"?
inputs/1?????????
p

 
? "???????????
)__inference_model_2_layer_call_fn_9296809?
$%1289j?g
`?]
S?P
*?'
inputs/0?????????@@
"?
inputs/1?????????
p 

 
? "???????????
%__inference_signature_wrapper_9296691?
$%1289y?v
? 
o?l
0
	input_act#? 
	input_act?????????
8
	input_img+?(
	input_img?????????@@"1?.
,
dense_3!?
dense_3?????????