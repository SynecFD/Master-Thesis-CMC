??
??
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
8
Const
output"dtype"
valuetensor"
dtypetype
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
-
Tanh
x"T
y"T"
Ttype:

2
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12unknown8??
p

hid/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:#*
shared_name
hid/kernel
i
hid/kernel/Read/ReadVariableOpReadVariableOp
hid/kernel*
_output_shapes

:#*
dtype0
h
hid/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
hid/bias
a
hid/bias/Read/ReadVariableOpReadVariableOphid/bias*
_output_shapes
:*
dtype0
r
out1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_nameout1/kernel
k
out1/kernel/Read/ReadVariableOpReadVariableOpout1/kernel*
_output_shapes

: *
dtype0
j
	out1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	out1/bias
c
out1/bias/Read/ReadVariableOpReadVariableOp	out1/bias*
_output_shapes
: *
dtype0
r
out2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_nameout2/kernel
k
out2/kernel/Read/ReadVariableOpReadVariableOpout2/kernel*
_output_shapes

:*
dtype0
j
	out2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name	out2/bias
c
out2/bias/Read/ReadVariableOpReadVariableOp	out2/bias*
_output_shapes
:*
dtype0
|
training_4/Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *%
shared_nametraining_4/Adam/iter
u
(training_4/Adam/iter/Read/ReadVariableOpReadVariableOptraining_4/Adam/iter*
_output_shapes
: *
dtype0	
?
training_4/Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nametraining_4/Adam/beta_1
y
*training_4/Adam/beta_1/Read/ReadVariableOpReadVariableOptraining_4/Adam/beta_1*
_output_shapes
: *
dtype0
?
training_4/Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nametraining_4/Adam/beta_2
y
*training_4/Adam/beta_2/Read/ReadVariableOpReadVariableOptraining_4/Adam/beta_2*
_output_shapes
: *
dtype0
~
training_4/Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nametraining_4/Adam/decay
w
)training_4/Adam/decay/Read/ReadVariableOpReadVariableOptraining_4/Adam/decay*
_output_shapes
: *
dtype0
?
training_4/Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_nametraining_4/Adam/learning_rate
?
1training_4/Adam/learning_rate/Read/ReadVariableOpReadVariableOptraining_4/Adam/learning_rate*
_output_shapes
: *
dtype0
?
training_4/Adam/hid/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:#*-
shared_nametraining_4/Adam/hid/kernel/m
?
0training_4/Adam/hid/kernel/m/Read/ReadVariableOpReadVariableOptraining_4/Adam/hid/kernel/m*
_output_shapes

:#*
dtype0
?
training_4/Adam/hid/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nametraining_4/Adam/hid/bias/m
?
.training_4/Adam/hid/bias/m/Read/ReadVariableOpReadVariableOptraining_4/Adam/hid/bias/m*
_output_shapes
:*
dtype0
?
training_4/Adam/out1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *.
shared_nametraining_4/Adam/out1/kernel/m
?
1training_4/Adam/out1/kernel/m/Read/ReadVariableOpReadVariableOptraining_4/Adam/out1/kernel/m*
_output_shapes

: *
dtype0
?
training_4/Adam/out1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nametraining_4/Adam/out1/bias/m
?
/training_4/Adam/out1/bias/m/Read/ReadVariableOpReadVariableOptraining_4/Adam/out1/bias/m*
_output_shapes
: *
dtype0
?
training_4/Adam/out2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nametraining_4/Adam/out2/kernel/m
?
1training_4/Adam/out2/kernel/m/Read/ReadVariableOpReadVariableOptraining_4/Adam/out2/kernel/m*
_output_shapes

:*
dtype0
?
training_4/Adam/out2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nametraining_4/Adam/out2/bias/m
?
/training_4/Adam/out2/bias/m/Read/ReadVariableOpReadVariableOptraining_4/Adam/out2/bias/m*
_output_shapes
:*
dtype0
?
training_4/Adam/hid/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:#*-
shared_nametraining_4/Adam/hid/kernel/v
?
0training_4/Adam/hid/kernel/v/Read/ReadVariableOpReadVariableOptraining_4/Adam/hid/kernel/v*
_output_shapes

:#*
dtype0
?
training_4/Adam/hid/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nametraining_4/Adam/hid/bias/v
?
.training_4/Adam/hid/bias/v/Read/ReadVariableOpReadVariableOptraining_4/Adam/hid/bias/v*
_output_shapes
:*
dtype0
?
training_4/Adam/out1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *.
shared_nametraining_4/Adam/out1/kernel/v
?
1training_4/Adam/out1/kernel/v/Read/ReadVariableOpReadVariableOptraining_4/Adam/out1/kernel/v*
_output_shapes

: *
dtype0
?
training_4/Adam/out1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nametraining_4/Adam/out1/bias/v
?
/training_4/Adam/out1/bias/v/Read/ReadVariableOpReadVariableOptraining_4/Adam/out1/bias/v*
_output_shapes
: *
dtype0
?
training_4/Adam/out2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nametraining_4/Adam/out2/kernel/v
?
1training_4/Adam/out2/kernel/v/Read/ReadVariableOpReadVariableOptraining_4/Adam/out2/kernel/v*
_output_shapes

:*
dtype0
?
training_4/Adam/out2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nametraining_4/Adam/out2/bias/v
?
/training_4/Adam/out2/bias/v/Read/ReadVariableOpReadVariableOptraining_4/Adam/out2/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?#
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?#
value?#B?# B?#
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
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
?

kernel
bias
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
?

kernel
bias
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
?

kernel
bias
#_self_saveable_object_factories
regularization_losses
	variables
 trainable_variables
!	keras_api
?
"iter

#beta_1

$beta_2
	%decay
&learning_ratem;m<m=m>m?m@vAvBvCvDvEvF
 
 
 
*
0
1
2
3
4
5
*
0
1
2
3
4
5
?
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
VT
VARIABLE_VALUE
hid/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEhid/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
?
,layer_metrics
-layer_regularization_losses
regularization_losses
.non_trainable_variables

/layers
	variables
0metrics
trainable_variables
WU
VARIABLE_VALUEout1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	out1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
?
1layer_metrics
2layer_regularization_losses
regularization_losses
3non_trainable_variables

4layers
	variables
5metrics
trainable_variables
WU
VARIABLE_VALUEout2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	out2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
?
6layer_metrics
7layer_regularization_losses
regularization_losses
8non_trainable_variables

9layers
	variables
:metrics
 trainable_variables
SQ
VARIABLE_VALUEtraining_4/Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEtraining_4/Adam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEtraining_4/Adam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEtraining_4/Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEtraining_4/Adam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

0
1
2
3
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
??
VARIABLE_VALUEtraining_4/Adam/hid/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEtraining_4/Adam/hid/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining_4/Adam/out1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEtraining_4/Adam/out1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining_4/Adam/out2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEtraining_4/Adam/out2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining_4/Adam/hid/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEtraining_4/Adam/hid/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining_4/Adam/out1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEtraining_4/Adam/out1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining_4/Adam/out2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEtraining_4/Adam/out2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_2Placeholder*'
_output_shapes
:?????????#*
dtype0*
shape:?????????#
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_2
hid/kernelhid/biasout2/kernel	out2/biasout1/kernel	out1/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:????????? :?????????*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8? *.
f)R'
%__inference_signature_wrapper_9298657
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamehid/kernel/Read/ReadVariableOphid/bias/Read/ReadVariableOpout1/kernel/Read/ReadVariableOpout1/bias/Read/ReadVariableOpout2/kernel/Read/ReadVariableOpout2/bias/Read/ReadVariableOp(training_4/Adam/iter/Read/ReadVariableOp*training_4/Adam/beta_1/Read/ReadVariableOp*training_4/Adam/beta_2/Read/ReadVariableOp)training_4/Adam/decay/Read/ReadVariableOp1training_4/Adam/learning_rate/Read/ReadVariableOp0training_4/Adam/hid/kernel/m/Read/ReadVariableOp.training_4/Adam/hid/bias/m/Read/ReadVariableOp1training_4/Adam/out1/kernel/m/Read/ReadVariableOp/training_4/Adam/out1/bias/m/Read/ReadVariableOp1training_4/Adam/out2/kernel/m/Read/ReadVariableOp/training_4/Adam/out2/bias/m/Read/ReadVariableOp0training_4/Adam/hid/kernel/v/Read/ReadVariableOp.training_4/Adam/hid/bias/v/Read/ReadVariableOp1training_4/Adam/out1/kernel/v/Read/ReadVariableOp/training_4/Adam/out1/bias/v/Read/ReadVariableOp1training_4/Adam/out2/kernel/v/Read/ReadVariableOp/training_4/Adam/out2/bias/v/Read/ReadVariableOpConst*$
Tin
2	*
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
 __inference__traced_save_9298882
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename
hid/kernelhid/biasout1/kernel	out1/biasout2/kernel	out2/biastraining_4/Adam/itertraining_4/Adam/beta_1training_4/Adam/beta_2training_4/Adam/decaytraining_4/Adam/learning_ratetraining_4/Adam/hid/kernel/mtraining_4/Adam/hid/bias/mtraining_4/Adam/out1/kernel/mtraining_4/Adam/out1/bias/mtraining_4/Adam/out2/kernel/mtraining_4/Adam/out2/bias/mtraining_4/Adam/hid/kernel/vtraining_4/Adam/hid/bias/vtraining_4/Adam/out1/kernel/vtraining_4/Adam/out1/bias/vtraining_4/Adam/out2/kernel/vtraining_4/Adam/out2/bias/v*#
Tin
2*
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
#__inference__traced_restore_9298961??
?	
?
A__inference_out1_layer_call_and_return_conditional_losses_9298764

inputs#
matmul_readvariableop_kernel_10"
biasadd_readvariableop_bias_10
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_kernel_10*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_10*
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
x
%__inference_hid_layer_call_fn_9298753

inputs
kernel_9

bias_9
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputskernel_9bias_9*
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
GPU2 *0J 8? *I
fDRB
@__inference_hid_layer_call_and_return_conditional_losses_92985132
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
?	
?
A__inference_out2_layer_call_and_return_conditional_losses_9298536

inputs#
matmul_readvariableop_kernel_11"
biasadd_readvariableop_bias_11
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_kernel_11*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_11*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
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
?9
?

 __inference__traced_save_9298882
file_prefix)
%savev2_hid_kernel_read_readvariableop'
#savev2_hid_bias_read_readvariableop*
&savev2_out1_kernel_read_readvariableop(
$savev2_out1_bias_read_readvariableop*
&savev2_out2_kernel_read_readvariableop(
$savev2_out2_bias_read_readvariableop3
/savev2_training_4_adam_iter_read_readvariableop	5
1savev2_training_4_adam_beta_1_read_readvariableop5
1savev2_training_4_adam_beta_2_read_readvariableop4
0savev2_training_4_adam_decay_read_readvariableop<
8savev2_training_4_adam_learning_rate_read_readvariableop;
7savev2_training_4_adam_hid_kernel_m_read_readvariableop9
5savev2_training_4_adam_hid_bias_m_read_readvariableop<
8savev2_training_4_adam_out1_kernel_m_read_readvariableop:
6savev2_training_4_adam_out1_bias_m_read_readvariableop<
8savev2_training_4_adam_out2_kernel_m_read_readvariableop:
6savev2_training_4_adam_out2_bias_m_read_readvariableop;
7savev2_training_4_adam_hid_kernel_v_read_readvariableop9
5savev2_training_4_adam_hid_bias_v_read_readvariableop<
8savev2_training_4_adam_out1_kernel_v_read_readvariableop:
6savev2_training_4_adam_out1_bias_v_read_readvariableop<
8savev2_training_4_adam_out2_kernel_v_read_readvariableop:
6savev2_training_4_adam_out2_bias_v_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0%savev2_hid_kernel_read_readvariableop#savev2_hid_bias_read_readvariableop&savev2_out1_kernel_read_readvariableop$savev2_out1_bias_read_readvariableop&savev2_out2_kernel_read_readvariableop$savev2_out2_bias_read_readvariableop/savev2_training_4_adam_iter_read_readvariableop1savev2_training_4_adam_beta_1_read_readvariableop1savev2_training_4_adam_beta_2_read_readvariableop0savev2_training_4_adam_decay_read_readvariableop8savev2_training_4_adam_learning_rate_read_readvariableop7savev2_training_4_adam_hid_kernel_m_read_readvariableop5savev2_training_4_adam_hid_bias_m_read_readvariableop8savev2_training_4_adam_out1_kernel_m_read_readvariableop6savev2_training_4_adam_out1_bias_m_read_readvariableop8savev2_training_4_adam_out2_kernel_m_read_readvariableop6savev2_training_4_adam_out2_bias_m_read_readvariableop7savev2_training_4_adam_hid_kernel_v_read_readvariableop5savev2_training_4_adam_hid_bias_v_read_readvariableop8savev2_training_4_adam_out1_kernel_v_read_readvariableop6savev2_training_4_adam_out1_bias_v_read_readvariableop8savev2_training_4_adam_out2_kernel_v_read_readvariableop6savev2_training_4_adam_out2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *&
dtypes
2	2
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :#:: : ::: : : : : :#:: : :::#:: : ::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:#: 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:#: 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:#: 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
?	
?
)__inference_model_6_layer_call_fn_9298642
input_2
kernel_9

bias_9
	kernel_11
bias_11
	kernel_10
bias_10
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2kernel_9bias_9	kernel_11bias_11	kernel_10bias_10*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:????????? :?????????*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_model_6_layer_call_and_return_conditional_losses_92986312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:?????????#::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????#
!
_user_specified_name	input_2
?	
?
@__inference_hid_layer_call_and_return_conditional_losses_9298513

inputs"
matmul_readvariableop_kernel_9!
biasadd_readvariableop_bias_9
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_kernel_9*
_output_shapes

:#*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_9*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
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
?	
?
A__inference_out2_layer_call_and_return_conditional_losses_9298782

inputs#
matmul_readvariableop_kernel_11"
biasadd_readvariableop_bias_11
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_kernel_11*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_11*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
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
?
?
D__inference_model_6_layer_call_and_return_conditional_losses_9298631

inputs
hid_kernel_9

hid_bias_9
out2_kernel_11
out2_bias_11
out1_kernel_10
out1_bias_10
identity

identity_1??hid/StatefulPartitionedCall?out1/StatefulPartitionedCall?out2/StatefulPartitionedCall?
hid/StatefulPartitionedCallStatefulPartitionedCallinputshid_kernel_9
hid_bias_9*
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
GPU2 *0J 8? *I
fDRB
@__inference_hid_layer_call_and_return_conditional_losses_92985132
hid/StatefulPartitionedCall?
out2/StatefulPartitionedCallStatefulPartitionedCall$hid/StatefulPartitionedCall:output:0out2_kernel_11out2_bias_11*
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
GPU2 *0J 8? *J
fERC
A__inference_out2_layer_call_and_return_conditional_losses_92985362
out2/StatefulPartitionedCall?
out1/StatefulPartitionedCallStatefulPartitionedCall$hid/StatefulPartitionedCall:output:0out1_kernel_10out1_bias_10*
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
GPU2 *0J 8? *J
fERC
A__inference_out1_layer_call_and_return_conditional_losses_92985592
out1/StatefulPartitionedCall?
IdentityIdentity%out1/StatefulPartitionedCall:output:0^hid/StatefulPartitionedCall^out1/StatefulPartitionedCall^out2/StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity%out2/StatefulPartitionedCall:output:0^hid/StatefulPartitionedCall^out1/StatefulPartitionedCall^out2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:?????????#::::::2:
hid/StatefulPartitionedCallhid/StatefulPartitionedCall2<
out1/StatefulPartitionedCallout1/StatefulPartitionedCall2<
out2/StatefulPartitionedCallout2/StatefulPartitionedCall:O K
'
_output_shapes
:?????????#
 
_user_specified_nameinputs
?	
?
%__inference_signature_wrapper_9298657
input_2
kernel_9

bias_9
	kernel_11
bias_11
	kernel_10
bias_10
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2kernel_9bias_9	kernel_11bias_11	kernel_10bias_10*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:????????? :?????????*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8? *+
f&R$
"__inference__wrapped_model_92984982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:?????????#::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????#
!
_user_specified_name	input_2
?	
?
)__inference_model_6_layer_call_fn_9298735

inputs
kernel_9

bias_9
	kernel_11
bias_11
	kernel_10
bias_10
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputskernel_9bias_9	kernel_11bias_11	kernel_10bias_10*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:????????? :?????????*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_model_6_layer_call_and_return_conditional_losses_92986312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:?????????#::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????#
 
_user_specified_nameinputs
?
?
D__inference_model_6_layer_call_and_return_conditional_losses_9298573
input_2
hid_kernel_9

hid_bias_9
out2_kernel_11
out2_bias_11
out1_kernel_10
out1_bias_10
identity

identity_1??hid/StatefulPartitionedCall?out1/StatefulPartitionedCall?out2/StatefulPartitionedCall?
hid/StatefulPartitionedCallStatefulPartitionedCallinput_2hid_kernel_9
hid_bias_9*
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
GPU2 *0J 8? *I
fDRB
@__inference_hid_layer_call_and_return_conditional_losses_92985132
hid/StatefulPartitionedCall?
out2/StatefulPartitionedCallStatefulPartitionedCall$hid/StatefulPartitionedCall:output:0out2_kernel_11out2_bias_11*
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
GPU2 *0J 8? *J
fERC
A__inference_out2_layer_call_and_return_conditional_losses_92985362
out2/StatefulPartitionedCall?
out1/StatefulPartitionedCallStatefulPartitionedCall$hid/StatefulPartitionedCall:output:0out1_kernel_10out1_bias_10*
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
GPU2 *0J 8? *J
fERC
A__inference_out1_layer_call_and_return_conditional_losses_92985592
out1/StatefulPartitionedCall?
IdentityIdentity%out1/StatefulPartitionedCall:output:0^hid/StatefulPartitionedCall^out1/StatefulPartitionedCall^out2/StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity%out2/StatefulPartitionedCall:output:0^hid/StatefulPartitionedCall^out1/StatefulPartitionedCall^out2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:?????????#::::::2:
hid/StatefulPartitionedCallhid/StatefulPartitionedCall2<
out1/StatefulPartitionedCallout1/StatefulPartitionedCall2<
out2/StatefulPartitionedCallout2/StatefulPartitionedCall:P L
'
_output_shapes
:?????????#
!
_user_specified_name	input_2
?
{
&__inference_out2_layer_call_fn_9298789

inputs
	kernel_11
bias_11
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs	kernel_11bias_11*
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
GPU2 *0J 8? *J
fERC
A__inference_out2_layer_call_and_return_conditional_losses_92985362
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
?e
?
#__inference__traced_restore_9298961
file_prefix
assignvariableop_hid_kernel
assignvariableop_1_hid_bias"
assignvariableop_2_out1_kernel 
assignvariableop_3_out1_bias"
assignvariableop_4_out2_kernel 
assignvariableop_5_out2_bias+
'assignvariableop_6_training_4_adam_iter-
)assignvariableop_7_training_4_adam_beta_1-
)assignvariableop_8_training_4_adam_beta_2,
(assignvariableop_9_training_4_adam_decay5
1assignvariableop_10_training_4_adam_learning_rate4
0assignvariableop_11_training_4_adam_hid_kernel_m2
.assignvariableop_12_training_4_adam_hid_bias_m5
1assignvariableop_13_training_4_adam_out1_kernel_m3
/assignvariableop_14_training_4_adam_out1_bias_m5
1assignvariableop_15_training_4_adam_out2_kernel_m3
/assignvariableop_16_training_4_adam_out2_bias_m4
0assignvariableop_17_training_4_adam_hid_kernel_v2
.assignvariableop_18_training_4_adam_hid_bias_v5
1assignvariableop_19_training_4_adam_out1_kernel_v3
/assignvariableop_20_training_4_adam_out1_bias_v5
1assignvariableop_21_training_4_adam_out2_kernel_v3
/assignvariableop_22_training_4_adam_out2_bias_v
identity_24??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*C
value:B8B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*t
_output_shapesb
`::::::::::::::::::::::::*&
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_hid_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_hid_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_out1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_out1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_out2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_out2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp'assignvariableop_6_training_4_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp)assignvariableop_7_training_4_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp)assignvariableop_8_training_4_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp(assignvariableop_9_training_4_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp1assignvariableop_10_training_4_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp0assignvariableop_11_training_4_adam_hid_kernel_mIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp.assignvariableop_12_training_4_adam_hid_bias_mIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp1assignvariableop_13_training_4_adam_out1_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp/assignvariableop_14_training_4_adam_out1_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp1assignvariableop_15_training_4_adam_out2_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp/assignvariableop_16_training_4_adam_out2_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp0assignvariableop_17_training_4_adam_hid_kernel_vIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp.assignvariableop_18_training_4_adam_hid_bias_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp1assignvariableop_19_training_4_adam_out1_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp/assignvariableop_20_training_4_adam_out1_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp1assignvariableop_21_training_4_adam_out2_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp/assignvariableop_22_training_4_adam_out2_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_229
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_23Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_23?
Identity_24IdentityIdentity_23:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_24"#
identity_24Identity_24:output:0*q
_input_shapes`
^: :::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222(
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
?
?
D__inference_model_6_layer_call_and_return_conditional_losses_9298587
input_2
hid_kernel_9

hid_bias_9
out2_kernel_11
out2_bias_11
out1_kernel_10
out1_bias_10
identity

identity_1??hid/StatefulPartitionedCall?out1/StatefulPartitionedCall?out2/StatefulPartitionedCall?
hid/StatefulPartitionedCallStatefulPartitionedCallinput_2hid_kernel_9
hid_bias_9*
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
GPU2 *0J 8? *I
fDRB
@__inference_hid_layer_call_and_return_conditional_losses_92985132
hid/StatefulPartitionedCall?
out2/StatefulPartitionedCallStatefulPartitionedCall$hid/StatefulPartitionedCall:output:0out2_kernel_11out2_bias_11*
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
GPU2 *0J 8? *J
fERC
A__inference_out2_layer_call_and_return_conditional_losses_92985362
out2/StatefulPartitionedCall?
out1/StatefulPartitionedCallStatefulPartitionedCall$hid/StatefulPartitionedCall:output:0out1_kernel_10out1_bias_10*
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
GPU2 *0J 8? *J
fERC
A__inference_out1_layer_call_and_return_conditional_losses_92985592
out1/StatefulPartitionedCall?
IdentityIdentity%out1/StatefulPartitionedCall:output:0^hid/StatefulPartitionedCall^out1/StatefulPartitionedCall^out2/StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity%out2/StatefulPartitionedCall:output:0^hid/StatefulPartitionedCall^out1/StatefulPartitionedCall^out2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:?????????#::::::2:
hid/StatefulPartitionedCallhid/StatefulPartitionedCall2<
out1/StatefulPartitionedCallout1/StatefulPartitionedCall2<
out2/StatefulPartitionedCallout2/StatefulPartitionedCall:P L
'
_output_shapes
:?????????#
!
_user_specified_name	input_2
?	
?
@__inference_hid_layer_call_and_return_conditional_losses_9298746

inputs"
matmul_readvariableop_kernel_9!
biasadd_readvariableop_bias_9
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_kernel_9*
_output_shapes

:#*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_9*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
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
?	
?
)__inference_model_6_layer_call_fn_9298615
input_2
kernel_9

bias_9
	kernel_11
bias_11
	kernel_10
bias_10
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2kernel_9bias_9	kernel_11bias_11	kernel_10bias_10*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:????????? :?????????*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_model_6_layer_call_and_return_conditional_losses_92986042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:?????????#::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????#
!
_user_specified_name	input_2
?
?
D__inference_model_6_layer_call_and_return_conditional_losses_9298709

inputs&
"hid_matmul_readvariableop_kernel_9%
!hid_biasadd_readvariableop_bias_9(
$out2_matmul_readvariableop_kernel_11'
#out2_biasadd_readvariableop_bias_11(
$out1_matmul_readvariableop_kernel_10'
#out1_biasadd_readvariableop_bias_10
identity

identity_1??hid/BiasAdd/ReadVariableOp?hid/MatMul/ReadVariableOp?out1/BiasAdd/ReadVariableOp?out1/MatMul/ReadVariableOp?out2/BiasAdd/ReadVariableOp?out2/MatMul/ReadVariableOp?
hid/MatMul/ReadVariableOpReadVariableOp"hid_matmul_readvariableop_kernel_9*
_output_shapes

:#*
dtype02
hid/MatMul/ReadVariableOp

hid/MatMulMatMulinputs!hid/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

hid/MatMul?
hid/BiasAdd/ReadVariableOpReadVariableOp!hid_biasadd_readvariableop_bias_9*
_output_shapes
:*
dtype02
hid/BiasAdd/ReadVariableOp?
hid/BiasAddBiasAddhid/MatMul:product:0"hid/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
hid/BiasAddd
hid/TanhTanhhid/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

hid/Tanh?
out2/MatMul/ReadVariableOpReadVariableOp$out2_matmul_readvariableop_kernel_11*
_output_shapes

:*
dtype02
out2/MatMul/ReadVariableOp?
out2/MatMulMatMulhid/Tanh:y:0"out2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
out2/MatMul?
out2/BiasAdd/ReadVariableOpReadVariableOp#out2_biasadd_readvariableop_bias_11*
_output_shapes
:*
dtype02
out2/BiasAdd/ReadVariableOp?
out2/BiasAddBiasAddout2/MatMul:product:0#out2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
out2/BiasAddg
	out2/TanhTanhout2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
	out2/Tanh?
out1/MatMul/ReadVariableOpReadVariableOp$out1_matmul_readvariableop_kernel_10*
_output_shapes

: *
dtype02
out1/MatMul/ReadVariableOp?
out1/MatMulMatMulhid/Tanh:y:0"out1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
out1/MatMul?
out1/BiasAdd/ReadVariableOpReadVariableOp#out1_biasadd_readvariableop_bias_10*
_output_shapes
: *
dtype02
out1/BiasAdd/ReadVariableOp?
out1/BiasAddBiasAddout1/MatMul:product:0#out1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
out1/BiasAddg
	out1/ReluReluout1/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
	out1/Relu?
IdentityIdentityout1/Relu:activations:0^hid/BiasAdd/ReadVariableOp^hid/MatMul/ReadVariableOp^out1/BiasAdd/ReadVariableOp^out1/MatMul/ReadVariableOp^out2/BiasAdd/ReadVariableOp^out2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identityout2/Tanh:y:0^hid/BiasAdd/ReadVariableOp^hid/MatMul/ReadVariableOp^out1/BiasAdd/ReadVariableOp^out1/MatMul/ReadVariableOp^out2/BiasAdd/ReadVariableOp^out2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:?????????#::::::28
hid/BiasAdd/ReadVariableOphid/BiasAdd/ReadVariableOp26
hid/MatMul/ReadVariableOphid/MatMul/ReadVariableOp2:
out1/BiasAdd/ReadVariableOpout1/BiasAdd/ReadVariableOp28
out1/MatMul/ReadVariableOpout1/MatMul/ReadVariableOp2:
out2/BiasAdd/ReadVariableOpout2/BiasAdd/ReadVariableOp28
out2/MatMul/ReadVariableOpout2/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????#
 
_user_specified_nameinputs
?	
?
)__inference_model_6_layer_call_fn_9298722

inputs
kernel_9

bias_9
	kernel_11
bias_11
	kernel_10
bias_10
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputskernel_9bias_9	kernel_11bias_11	kernel_10bias_10*
Tin
	2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:????????? :?????????*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_model_6_layer_call_and_return_conditional_losses_92986042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:?????????#::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????#
 
_user_specified_nameinputs
?"
?
"__inference__wrapped_model_9298498
input_2.
*model_6_hid_matmul_readvariableop_kernel_9-
)model_6_hid_biasadd_readvariableop_bias_90
,model_6_out2_matmul_readvariableop_kernel_11/
+model_6_out2_biasadd_readvariableop_bias_110
,model_6_out1_matmul_readvariableop_kernel_10/
+model_6_out1_biasadd_readvariableop_bias_10
identity

identity_1??"model_6/hid/BiasAdd/ReadVariableOp?!model_6/hid/MatMul/ReadVariableOp?#model_6/out1/BiasAdd/ReadVariableOp?"model_6/out1/MatMul/ReadVariableOp?#model_6/out2/BiasAdd/ReadVariableOp?"model_6/out2/MatMul/ReadVariableOp?
!model_6/hid/MatMul/ReadVariableOpReadVariableOp*model_6_hid_matmul_readvariableop_kernel_9*
_output_shapes

:#*
dtype02#
!model_6/hid/MatMul/ReadVariableOp?
model_6/hid/MatMulMatMulinput_2)model_6/hid/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_6/hid/MatMul?
"model_6/hid/BiasAdd/ReadVariableOpReadVariableOp)model_6_hid_biasadd_readvariableop_bias_9*
_output_shapes
:*
dtype02$
"model_6/hid/BiasAdd/ReadVariableOp?
model_6/hid/BiasAddBiasAddmodel_6/hid/MatMul:product:0*model_6/hid/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_6/hid/BiasAdd|
model_6/hid/TanhTanhmodel_6/hid/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_6/hid/Tanh?
"model_6/out2/MatMul/ReadVariableOpReadVariableOp,model_6_out2_matmul_readvariableop_kernel_11*
_output_shapes

:*
dtype02$
"model_6/out2/MatMul/ReadVariableOp?
model_6/out2/MatMulMatMulmodel_6/hid/Tanh:y:0*model_6/out2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_6/out2/MatMul?
#model_6/out2/BiasAdd/ReadVariableOpReadVariableOp+model_6_out2_biasadd_readvariableop_bias_11*
_output_shapes
:*
dtype02%
#model_6/out2/BiasAdd/ReadVariableOp?
model_6/out2/BiasAddBiasAddmodel_6/out2/MatMul:product:0+model_6/out2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_6/out2/BiasAdd
model_6/out2/TanhTanhmodel_6/out2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_6/out2/Tanh?
"model_6/out1/MatMul/ReadVariableOpReadVariableOp,model_6_out1_matmul_readvariableop_kernel_10*
_output_shapes

: *
dtype02$
"model_6/out1/MatMul/ReadVariableOp?
model_6/out1/MatMulMatMulmodel_6/hid/Tanh:y:0*model_6/out1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
model_6/out1/MatMul?
#model_6/out1/BiasAdd/ReadVariableOpReadVariableOp+model_6_out1_biasadd_readvariableop_bias_10*
_output_shapes
: *
dtype02%
#model_6/out1/BiasAdd/ReadVariableOp?
model_6/out1/BiasAddBiasAddmodel_6/out1/MatMul:product:0+model_6/out1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
model_6/out1/BiasAdd
model_6/out1/ReluRelumodel_6/out1/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
model_6/out1/Relu?
IdentityIdentitymodel_6/out1/Relu:activations:0#^model_6/hid/BiasAdd/ReadVariableOp"^model_6/hid/MatMul/ReadVariableOp$^model_6/out1/BiasAdd/ReadVariableOp#^model_6/out1/MatMul/ReadVariableOp$^model_6/out2/BiasAdd/ReadVariableOp#^model_6/out2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identitymodel_6/out2/Tanh:y:0#^model_6/hid/BiasAdd/ReadVariableOp"^model_6/hid/MatMul/ReadVariableOp$^model_6/out1/BiasAdd/ReadVariableOp#^model_6/out1/MatMul/ReadVariableOp$^model_6/out2/BiasAdd/ReadVariableOp#^model_6/out2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:?????????#::::::2H
"model_6/hid/BiasAdd/ReadVariableOp"model_6/hid/BiasAdd/ReadVariableOp2F
!model_6/hid/MatMul/ReadVariableOp!model_6/hid/MatMul/ReadVariableOp2J
#model_6/out1/BiasAdd/ReadVariableOp#model_6/out1/BiasAdd/ReadVariableOp2H
"model_6/out1/MatMul/ReadVariableOp"model_6/out1/MatMul/ReadVariableOp2J
#model_6/out2/BiasAdd/ReadVariableOp#model_6/out2/BiasAdd/ReadVariableOp2H
"model_6/out2/MatMul/ReadVariableOp"model_6/out2/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????#
!
_user_specified_name	input_2
?
{
&__inference_out1_layer_call_fn_9298771

inputs
	kernel_10
bias_10
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs	kernel_10bias_10*
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
GPU2 *0J 8? *J
fERC
A__inference_out1_layer_call_and_return_conditional_losses_92985592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
D__inference_model_6_layer_call_and_return_conditional_losses_9298604

inputs
hid_kernel_9

hid_bias_9
out2_kernel_11
out2_bias_11
out1_kernel_10
out1_bias_10
identity

identity_1??hid/StatefulPartitionedCall?out1/StatefulPartitionedCall?out2/StatefulPartitionedCall?
hid/StatefulPartitionedCallStatefulPartitionedCallinputshid_kernel_9
hid_bias_9*
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
GPU2 *0J 8? *I
fDRB
@__inference_hid_layer_call_and_return_conditional_losses_92985132
hid/StatefulPartitionedCall?
out2/StatefulPartitionedCallStatefulPartitionedCall$hid/StatefulPartitionedCall:output:0out2_kernel_11out2_bias_11*
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
GPU2 *0J 8? *J
fERC
A__inference_out2_layer_call_and_return_conditional_losses_92985362
out2/StatefulPartitionedCall?
out1/StatefulPartitionedCallStatefulPartitionedCall$hid/StatefulPartitionedCall:output:0out1_kernel_10out1_bias_10*
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
GPU2 *0J 8? *J
fERC
A__inference_out1_layer_call_and_return_conditional_losses_92985592
out1/StatefulPartitionedCall?
IdentityIdentity%out1/StatefulPartitionedCall:output:0^hid/StatefulPartitionedCall^out1/StatefulPartitionedCall^out2/StatefulPartitionedCall*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identity%out2/StatefulPartitionedCall:output:0^hid/StatefulPartitionedCall^out1/StatefulPartitionedCall^out2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:?????????#::::::2:
hid/StatefulPartitionedCallhid/StatefulPartitionedCall2<
out1/StatefulPartitionedCallout1/StatefulPartitionedCall2<
out2/StatefulPartitionedCallout2/StatefulPartitionedCall:O K
'
_output_shapes
:?????????#
 
_user_specified_nameinputs
?
?
D__inference_model_6_layer_call_and_return_conditional_losses_9298683

inputs&
"hid_matmul_readvariableop_kernel_9%
!hid_biasadd_readvariableop_bias_9(
$out2_matmul_readvariableop_kernel_11'
#out2_biasadd_readvariableop_bias_11(
$out1_matmul_readvariableop_kernel_10'
#out1_biasadd_readvariableop_bias_10
identity

identity_1??hid/BiasAdd/ReadVariableOp?hid/MatMul/ReadVariableOp?out1/BiasAdd/ReadVariableOp?out1/MatMul/ReadVariableOp?out2/BiasAdd/ReadVariableOp?out2/MatMul/ReadVariableOp?
hid/MatMul/ReadVariableOpReadVariableOp"hid_matmul_readvariableop_kernel_9*
_output_shapes

:#*
dtype02
hid/MatMul/ReadVariableOp

hid/MatMulMatMulinputs!hid/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2

hid/MatMul?
hid/BiasAdd/ReadVariableOpReadVariableOp!hid_biasadd_readvariableop_bias_9*
_output_shapes
:*
dtype02
hid/BiasAdd/ReadVariableOp?
hid/BiasAddBiasAddhid/MatMul:product:0"hid/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
hid/BiasAddd
hid/TanhTanhhid/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2

hid/Tanh?
out2/MatMul/ReadVariableOpReadVariableOp$out2_matmul_readvariableop_kernel_11*
_output_shapes

:*
dtype02
out2/MatMul/ReadVariableOp?
out2/MatMulMatMulhid/Tanh:y:0"out2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
out2/MatMul?
out2/BiasAdd/ReadVariableOpReadVariableOp#out2_biasadd_readvariableop_bias_11*
_output_shapes
:*
dtype02
out2/BiasAdd/ReadVariableOp?
out2/BiasAddBiasAddout2/MatMul:product:0#out2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
out2/BiasAddg
	out2/TanhTanhout2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
	out2/Tanh?
out1/MatMul/ReadVariableOpReadVariableOp$out1_matmul_readvariableop_kernel_10*
_output_shapes

: *
dtype02
out1/MatMul/ReadVariableOp?
out1/MatMulMatMulhid/Tanh:y:0"out1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
out1/MatMul?
out1/BiasAdd/ReadVariableOpReadVariableOp#out1_biasadd_readvariableop_bias_10*
_output_shapes
: *
dtype02
out1/BiasAdd/ReadVariableOp?
out1/BiasAddBiasAddout1/MatMul:product:0#out1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
out1/BiasAddg
	out1/ReluReluout1/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
	out1/Relu?
IdentityIdentityout1/Relu:activations:0^hid/BiasAdd/ReadVariableOp^hid/MatMul/ReadVariableOp^out1/BiasAdd/ReadVariableOp^out1/MatMul/ReadVariableOp^out2/BiasAdd/ReadVariableOp^out2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:????????? 2

Identity?

Identity_1Identityout2/Tanh:y:0^hid/BiasAdd/ReadVariableOp^hid/MatMul/ReadVariableOp^out1/BiasAdd/ReadVariableOp^out1/MatMul/ReadVariableOp^out2/BiasAdd/ReadVariableOp^out2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*>
_input_shapes-
+:?????????#::::::28
hid/BiasAdd/ReadVariableOphid/BiasAdd/ReadVariableOp26
hid/MatMul/ReadVariableOphid/MatMul/ReadVariableOp2:
out1/BiasAdd/ReadVariableOpout1/BiasAdd/ReadVariableOp28
out1/MatMul/ReadVariableOpout1/MatMul/ReadVariableOp2:
out2/BiasAdd/ReadVariableOpout2/BiasAdd/ReadVariableOp28
out2/MatMul/ReadVariableOpout2/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????#
 
_user_specified_nameinputs
?	
?
A__inference_out1_layer_call_and_return_conditional_losses_9298559

inputs#
matmul_readvariableop_kernel_10"
biasadd_readvariableop_bias_10
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_kernel_10*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_10*
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
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_20
serving_default_input_2:0?????????#8
out10
StatefulPartitionedCall:0????????? 8
out20
StatefulPartitionedCall:1?????????tensorflow/serving/predict:??
?&
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer

signatures
#_self_saveable_object_factories
regularization_losses
		variables

trainable_variables
	keras_api
G__call__
*H&call_and_return_all_conditional_losses
I_default_save_signature"?#
_tf_keras_network?#{"class_name": "Functional", "name": "model_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 35]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "hid", "trainable": true, "dtype": "float32", "units": 20, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hid", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "out1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "out1", "inbound_nodes": [[["hid", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "out2", "trainable": true, "dtype": "float32", "units": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "out2", "inbound_nodes": [[["hid", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["out1", 0, 0], ["out2", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 35]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 35]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 35]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "hid", "trainable": true, "dtype": "float32", "units": 20, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "hid", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "out1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "out1", "inbound_nodes": [[["hid", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "out2", "trainable": true, "dtype": "float32", "units": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "out2", "inbound_nodes": [[["hid", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["out1", 0, 0], ["out2", 0, 0]]}}, "training_config": {"loss": ["mean_squared_error", "mean_squared_error"], "metrics": [], "loss_weights": null, "sample_weight_mode": null, "weighted_metrics": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
#_self_saveable_object_factories"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 35]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 35]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
?

kernel
bias
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
J__call__
*K&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "hid", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "hid", "trainable": true, "dtype": "float32", "units": 20, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 35}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35]}}
?

kernel
bias
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
L__call__
*M&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "out1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "out1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20]}}
?

kernel
bias
#_self_saveable_object_factories
regularization_losses
	variables
 trainable_variables
!	keras_api
N__call__
*O&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "out2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "out2", "trainable": true, "dtype": "float32", "units": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20]}}
?
"iter

#beta_1

$beta_2
	%decay
&learning_ratem;m<m=m>m?m@vAvBvCvDvEvF"
	optimizer
,
Pserving_default"
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
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
?
'layer_metrics
regularization_losses
(layer_regularization_losses
)non_trainable_variables

*layers
		variables
+metrics

trainable_variables
G__call__
I_default_save_signature
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
:#2
hid/kernel
:2hid/bias
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
?
,layer_metrics
-layer_regularization_losses
regularization_losses
.non_trainable_variables

/layers
	variables
0metrics
trainable_variables
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
: 2out1/kernel
: 2	out1/bias
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
?
1layer_metrics
2layer_regularization_losses
regularization_losses
3non_trainable_variables

4layers
	variables
5metrics
trainable_variables
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
:2out2/kernel
:2	out2/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
6layer_metrics
7layer_regularization_losses
regularization_losses
8non_trainable_variables

9layers
	variables
:metrics
 trainable_variables
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
:	 (2training_4/Adam/iter
 : (2training_4/Adam/beta_1
 : (2training_4/Adam/beta_2
: (2training_4/Adam/decay
':% (2training_4/Adam/learning_rate
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
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
,:*#2training_4/Adam/hid/kernel/m
&:$2training_4/Adam/hid/bias/m
-:+ 2training_4/Adam/out1/kernel/m
':% 2training_4/Adam/out1/bias/m
-:+2training_4/Adam/out2/kernel/m
':%2training_4/Adam/out2/bias/m
,:*#2training_4/Adam/hid/kernel/v
&:$2training_4/Adam/hid/bias/v
-:+ 2training_4/Adam/out1/kernel/v
':% 2training_4/Adam/out1/bias/v
-:+2training_4/Adam/out2/kernel/v
':%2training_4/Adam/out2/bias/v
?2?
)__inference_model_6_layer_call_fn_9298722
)__inference_model_6_layer_call_fn_9298735
)__inference_model_6_layer_call_fn_9298615
)__inference_model_6_layer_call_fn_9298642?
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
D__inference_model_6_layer_call_and_return_conditional_losses_9298683
D__inference_model_6_layer_call_and_return_conditional_losses_9298587
D__inference_model_6_layer_call_and_return_conditional_losses_9298709
D__inference_model_6_layer_call_and_return_conditional_losses_9298573?
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
?2?
"__inference__wrapped_model_9298498?
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
annotations? *&?#
!?
input_2?????????#
?2?
%__inference_hid_layer_call_fn_9298753?
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
@__inference_hid_layer_call_and_return_conditional_losses_9298746?
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
&__inference_out1_layer_call_fn_9298771?
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
A__inference_out1_layer_call_and_return_conditional_losses_9298764?
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
&__inference_out2_layer_call_fn_9298789?
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
A__inference_out2_layer_call_and_return_conditional_losses_9298782?
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
%__inference_signature_wrapper_9298657input_2"?
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
"__inference__wrapped_model_9298498?0?-
&?#
!?
input_2?????????#
? "S?P
&
out1?
out1????????? 
&
out2?
out2??????????
@__inference_hid_layer_call_and_return_conditional_losses_9298746\/?,
%?"
 ?
inputs?????????#
? "%?"
?
0?????????
? x
%__inference_hid_layer_call_fn_9298753O/?,
%?"
 ?
inputs?????????#
? "???????????
D__inference_model_6_layer_call_and_return_conditional_losses_9298573?8?5
.?+
!?
input_2?????????#
p

 
? "K?H
A?>
?
0/0????????? 
?
0/1?????????
? ?
D__inference_model_6_layer_call_and_return_conditional_losses_9298587?8?5
.?+
!?
input_2?????????#
p 

 
? "K?H
A?>
?
0/0????????? 
?
0/1?????????
? ?
D__inference_model_6_layer_call_and_return_conditional_losses_9298683?7?4
-?*
 ?
inputs?????????#
p

 
? "K?H
A?>
?
0/0????????? 
?
0/1?????????
? ?
D__inference_model_6_layer_call_and_return_conditional_losses_9298709?7?4
-?*
 ?
inputs?????????#
p 

 
? "K?H
A?>
?
0/0????????? 
?
0/1?????????
? ?
)__inference_model_6_layer_call_fn_9298615?8?5
.?+
!?
input_2?????????#
p

 
? "=?:
?
0????????? 
?
1??????????
)__inference_model_6_layer_call_fn_9298642?8?5
.?+
!?
input_2?????????#
p 

 
? "=?:
?
0????????? 
?
1??????????
)__inference_model_6_layer_call_fn_9298722?7?4
-?*
 ?
inputs?????????#
p

 
? "=?:
?
0????????? 
?
1??????????
)__inference_model_6_layer_call_fn_9298735?7?4
-?*
 ?
inputs?????????#
p 

 
? "=?:
?
0????????? 
?
1??????????
A__inference_out1_layer_call_and_return_conditional_losses_9298764\/?,
%?"
 ?
inputs?????????
? "%?"
?
0????????? 
? y
&__inference_out1_layer_call_fn_9298771O/?,
%?"
 ?
inputs?????????
? "?????????? ?
A__inference_out2_layer_call_and_return_conditional_losses_9298782\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? y
&__inference_out2_layer_call_fn_9298789O/?,
%?"
 ?
inputs?????????
? "???????????
%__inference_signature_wrapper_9298657?;?8
? 
1?.
,
input_2!?
input_2?????????#"S?P
&
out1?
out1????????? 
&
out2?
out2?????????