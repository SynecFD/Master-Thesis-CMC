??
??
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
=
Mul
x"T
y"T
z"T"
Ttype:
2	?
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
?
ResizeNearestNeighbor
images"T
size
resized_images"T"
Ttype:
2
	"
align_cornersbool( "
half_pixel_centersbool( 
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
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
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
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
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
 ?"serve*2.4.12unknown8??
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
w
dense2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*
shared_namedense2/kernel
p
!dense2/kernel/Read/ReadVariableOpReadVariableOpdense2/kernel*
_output_shapes
:	 ?*
dtype0
o
dense2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense2/bias
h
dense2/bias/Read/ReadVariableOpReadVariableOpdense2/bias*
_output_shapes	
:?*
dtype0
?
dec_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namedec_conv1/kernel
}
$dec_conv1/kernel/Read/ReadVariableOpReadVariableOpdec_conv1/kernel*&
_output_shapes
:*
dtype0
t
dec_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedec_conv1/bias
m
"dec_conv1/bias/Read/ReadVariableOpReadVariableOpdec_conv1/bias*
_output_shapes
:*
dtype0
?
dec_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*!
shared_namedec_conv2/kernel
}
$dec_conv2/kernel/Read/ReadVariableOpReadVariableOpdec_conv2/kernel*&
_output_shapes
:
*
dtype0
t
dec_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedec_conv2/bias
m
"dec_conv2/bias/Read/ReadVariableOpReadVariableOpdec_conv2/bias*
_output_shapes
:*
dtype0
?
dec_conv3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namedec_conv3/kernel
}
$dec_conv3/kernel/Read/ReadVariableOpReadVariableOpdec_conv3/kernel*&
_output_shapes
:*
dtype0
t
dec_conv3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedec_conv3/bias
m
"dec_conv3/bias/Read/ReadVariableOpReadVariableOpdec_conv3/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?0
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?/
value?/B?/ B?/
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11

signatures
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
%
#_self_saveable_object_factories
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
w
#"_self_saveable_object_factories
#regularization_losses
$	variables
%trainable_variables
&	keras_api
?

'kernel
(bias
#)_self_saveable_object_factories
*regularization_losses
+	variables
,trainable_variables
-	keras_api
?

.kernel
/bias
#0_self_saveable_object_factories
1regularization_losses
2	variables
3trainable_variables
4	keras_api
w
#5_self_saveable_object_factories
6regularization_losses
7	variables
8trainable_variables
9	keras_api
?

:kernel
;bias
#<_self_saveable_object_factories
=regularization_losses
>	variables
?trainable_variables
@	keras_api
w
#A_self_saveable_object_factories
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
?

Fkernel
Gbias
#H_self_saveable_object_factories
Iregularization_losses
J	variables
Ktrainable_variables
L	keras_api
w
#M_self_saveable_object_factories
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
?

Rkernel
Sbias
#T_self_saveable_object_factories
Uregularization_losses
V	variables
Wtrainable_variables
X	keras_api
 
 
 
f
0
1
2
3
'4
(5
.6
/7
:8
;9
F10
G11
R12
S13
f
0
1
2
3
'4
(5
.6
/7
:8
;9
F10
G11
R12
S13
?
Ylayer_metrics
regularization_losses
Zlayer_regularization_losses
[non_trainable_variables

\layers
	variables
]metrics
trainable_variables
 
XV
VARIABLE_VALUEconv1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
?
^layer_metrics
_layer_regularization_losses
regularization_losses
`non_trainable_variables

alayers
	variables
bmetrics
trainable_variables
XV
VARIABLE_VALUEconv2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
?
clayer_metrics
dlayer_regularization_losses
regularization_losses
enon_trainable_variables

flayers
	variables
gmetrics
 trainable_variables
 
 
 
 
?
hlayer_metrics
ilayer_regularization_losses
#regularization_losses
jnon_trainable_variables

klayers
$	variables
lmetrics
%trainable_variables
YW
VARIABLE_VALUEdense1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

'0
(1

'0
(1
?
mlayer_metrics
nlayer_regularization_losses
*regularization_losses
onon_trainable_variables

players
+	variables
qmetrics
,trainable_variables
YW
VARIABLE_VALUEdense2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

.0
/1

.0
/1
?
rlayer_metrics
slayer_regularization_losses
1regularization_losses
tnon_trainable_variables

ulayers
2	variables
vmetrics
3trainable_variables
 
 
 
 
?
wlayer_metrics
xlayer_regularization_losses
6regularization_losses
ynon_trainable_variables

zlayers
7	variables
{metrics
8trainable_variables
\Z
VARIABLE_VALUEdec_conv1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdec_conv1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

:0
;1

:0
;1
?
|layer_metrics
}layer_regularization_losses
=regularization_losses
~non_trainable_variables

layers
>	variables
?metrics
?trainable_variables
 
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
Bregularization_losses
?non_trainable_variables
?layers
C	variables
?metrics
Dtrainable_variables
\Z
VARIABLE_VALUEdec_conv2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdec_conv2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

F0
G1

F0
G1
?
?layer_metrics
 ?layer_regularization_losses
Iregularization_losses
?non_trainable_variables
?layers
J	variables
?metrics
Ktrainable_variables
 
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
Nregularization_losses
?non_trainable_variables
?layers
O	variables
?metrics
Ptrainable_variables
\Z
VARIABLE_VALUEdec_conv3/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdec_conv3/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

R0
S1

R0
S1
?
?layer_metrics
 ?layer_regularization_losses
Uregularization_losses
?non_trainable_variables
?layers
V	variables
?metrics
Wtrainable_variables
 
 
 
V
0
1
2
3
4
5
6
7
	8

9
10
11
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
?
serving_default_input_imgPlaceholder*/
_output_shapes
:?????????@@*
dtype0*$
shape:?????????@@
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_imgconv1/kernel
conv1/biasconv2/kernel
conv2/biasdense1/kerneldense1/biasdense2/kerneldense2/biasdec_conv1/kerneldec_conv1/biasdec_conv2/kerneldec_conv2/biasdec_conv3/kerneldec_conv3/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8? *.
f)R'
%__inference_signature_wrapper_9297536
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename conv1/kernel/Read/ReadVariableOpconv1/bias/Read/ReadVariableOp conv2/kernel/Read/ReadVariableOpconv2/bias/Read/ReadVariableOp!dense1/kernel/Read/ReadVariableOpdense1/bias/Read/ReadVariableOp!dense2/kernel/Read/ReadVariableOpdense2/bias/Read/ReadVariableOp$dec_conv1/kernel/Read/ReadVariableOp"dec_conv1/bias/Read/ReadVariableOp$dec_conv2/kernel/Read/ReadVariableOp"dec_conv2/bias/Read/ReadVariableOp$dec_conv3/kernel/Read/ReadVariableOp"dec_conv3/bias/Read/ReadVariableOpConst*
Tin
2*
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
 __inference__traced_save_9297957
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1/kernel
conv1/biasconv2/kernel
conv2/biasdense1/kerneldense1/biasdense2/kerneldense2/biasdec_conv1/kerneldec_conv1/biasdec_conv2/kerneldec_conv2/biasdec_conv3/kerneldec_conv3/bias*
Tin
2*
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
#__inference__traced_restore_9298009??
?
E
)__inference_reshape_layer_call_fn_9297838

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_92972992
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_dec_conv3_layer_call_fn_9297892

inputs
	kernel_26
bias_26
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs	kernel_26bias_26*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dec_conv3_layer_call_and_return_conditional_losses_92973762
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
C__inference_dense1_layer_call_and_return_conditional_losses_9297794

inputs#
matmul_readvariableop_kernel_22"
biasadd_readvariableop_bias_22
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_kernel_22*
_output_shapes
:	? *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_22*
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
?2
?
D__inference_model_4_layer_call_and_return_conditional_losses_9297389
	input_img
conv1_kernel_20
conv1_bias_20
conv2_kernel_21
conv2_bias_21
dense1_kernel_22
dense1_bias_22
dense2_kernel_23
dense2_bias_23
dec_conv1_kernel_24
dec_conv1_bias_24
dec_conv2_kernel_25
dec_conv2_bias_25
dec_conv3_kernel_26
dec_conv3_bias_26
identity??conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?!dec_conv1/StatefulPartitionedCall?!dec_conv2/StatefulPartitionedCall?!dec_conv3/StatefulPartitionedCall?dense1/StatefulPartitionedCall?dense2/StatefulPartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCall	input_imgconv1_kernel_20conv1_bias_20*
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
B__inference_conv1_layer_call_and_return_conditional_losses_92971902
conv1/StatefulPartitionedCall?
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0conv2_kernel_21conv2_bias_21*
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
B__inference_conv2_layer_call_and_return_conditional_losses_92972132
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
D__inference_flatten_layer_call_and_return_conditional_losses_92972312
flatten/PartitionedCall?
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_kernel_22dense1_bias_22*
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
C__inference_dense1_layer_call_and_return_conditional_losses_92972502 
dense1/StatefulPartitionedCall?
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_kernel_23dense2_bias_23*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_dense2_layer_call_and_return_conditional_losses_92972732 
dense2/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall'dense2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_92972992
reshape/PartitionedCall?
!dec_conv1/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0dec_conv1_kernel_24dec_conv1_bias_24*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dec_conv1_layer_call_and_return_conditional_losses_92973182#
!dec_conv1/StatefulPartitionedCall?
up_sampling2d/PartitionedCallPartitionedCall*dec_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_92971412
up_sampling2d/PartitionedCall?
!dec_conv2/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0dec_conv2_kernel_25dec_conv2_bias_25*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dec_conv2_layer_call_and_return_conditional_losses_92973472#
!dec_conv2/StatefulPartitionedCall?
up_sampling2d_1/PartitionedCallPartitionedCall*dec_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_92971722!
up_sampling2d_1/PartitionedCall?
!dec_conv3/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0dec_conv3_kernel_26dec_conv3_bias_26*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dec_conv3_layer_call_and_return_conditional_losses_92973762#
!dec_conv3/StatefulPartitionedCall?
IdentityIdentity*dec_conv3/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall"^dec_conv1/StatefulPartitionedCall"^dec_conv2/StatefulPartitionedCall"^dec_conv3/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????@@::::::::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2F
!dec_conv1/StatefulPartitionedCall!dec_conv1/StatefulPartitionedCall2F
!dec_conv2/StatefulPartitionedCall!dec_conv2/StatefulPartitionedCall2F
!dec_conv3/StatefulPartitionedCall!dec_conv3/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall:Z V
/
_output_shapes
:?????????@@
#
_user_specified_name	input_img
?
|
'__inference_conv1_layer_call_fn_9297754

inputs
	kernel_20
bias_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs	kernel_20bias_20*
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
B__inference_conv1_layer_call_and_return_conditional_losses_92971902
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
F__inference_dec_conv3_layer_call_and_return_conditional_losses_9297885

inputs#
conv2d_readvariableop_kernel_26"
biasadd_readvariableop_bias_26
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_kernel_26*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_26*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?2
?
D__inference_model_4_layer_call_and_return_conditional_losses_9297498

inputs
conv1_kernel_20
conv1_bias_20
conv2_kernel_21
conv2_bias_21
dense1_kernel_22
dense1_bias_22
dense2_kernel_23
dense2_bias_23
dec_conv1_kernel_24
dec_conv1_bias_24
dec_conv2_kernel_25
dec_conv2_bias_25
dec_conv3_kernel_26
dec_conv3_bias_26
identity??conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?!dec_conv1/StatefulPartitionedCall?!dec_conv2/StatefulPartitionedCall?!dec_conv3/StatefulPartitionedCall?dense1/StatefulPartitionedCall?dense2/StatefulPartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_kernel_20conv1_bias_20*
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
B__inference_conv1_layer_call_and_return_conditional_losses_92971902
conv1/StatefulPartitionedCall?
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0conv2_kernel_21conv2_bias_21*
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
B__inference_conv2_layer_call_and_return_conditional_losses_92972132
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
D__inference_flatten_layer_call_and_return_conditional_losses_92972312
flatten/PartitionedCall?
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_kernel_22dense1_bias_22*
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
C__inference_dense1_layer_call_and_return_conditional_losses_92972502 
dense1/StatefulPartitionedCall?
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_kernel_23dense2_bias_23*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_dense2_layer_call_and_return_conditional_losses_92972732 
dense2/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall'dense2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_92972992
reshape/PartitionedCall?
!dec_conv1/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0dec_conv1_kernel_24dec_conv1_bias_24*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dec_conv1_layer_call_and_return_conditional_losses_92973182#
!dec_conv1/StatefulPartitionedCall?
up_sampling2d/PartitionedCallPartitionedCall*dec_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_92971412
up_sampling2d/PartitionedCall?
!dec_conv2/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0dec_conv2_kernel_25dec_conv2_bias_25*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dec_conv2_layer_call_and_return_conditional_losses_92973472#
!dec_conv2/StatefulPartitionedCall?
up_sampling2d_1/PartitionedCallPartitionedCall*dec_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_92971722!
up_sampling2d_1/PartitionedCall?
!dec_conv3/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0dec_conv3_kernel_26dec_conv3_bias_26*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dec_conv3_layer_call_and_return_conditional_losses_92973762#
!dec_conv3/StatefulPartitionedCall?
IdentityIdentity*dec_conv3/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall"^dec_conv1/StatefulPartitionedCall"^dec_conv2/StatefulPartitionedCall"^dec_conv3/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????@@::::::::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2F
!dec_conv1/StatefulPartitionedCall!dec_conv1/StatefulPartitionedCall2F
!dec_conv2/StatefulPartitionedCall!dec_conv2/StatefulPartitionedCall2F
!dec_conv3/StatefulPartitionedCall!dec_conv3/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
K
/__inference_up_sampling2d_layer_call_fn_9297144

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_92971412
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?<
?
#__inference__traced_restore_9298009
file_prefix!
assignvariableop_conv1_kernel!
assignvariableop_1_conv1_bias#
assignvariableop_2_conv2_kernel!
assignvariableop_3_conv2_bias$
 assignvariableop_4_dense1_kernel"
assignvariableop_5_dense1_bias$
 assignvariableop_6_dense2_kernel"
assignvariableop_7_dense2_bias'
#assignvariableop_8_dec_conv1_kernel%
!assignvariableop_9_dec_conv1_bias(
$assignvariableop_10_dec_conv2_kernel&
"assignvariableop_11_dec_conv2_bias(
$assignvariableop_12_dec_conv3_kernel&
"assignvariableop_13_dec_conv3_bias
identity_15??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*P
_output_shapes>
<:::::::::::::::*
dtypes
22
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
AssignVariableOp_6AssignVariableOp assignvariableop_6_dense2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp#assignvariableop_8_dec_conv1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp!assignvariableop_9_dec_conv1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp$assignvariableop_10_dec_conv2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp"assignvariableop_11_dec_conv2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp$assignvariableop_12_dec_conv3_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp"assignvariableop_13_dec_conv3_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_139
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_14Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_14?
Identity_15IdentityIdentity_14:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_15"#
identity_15Identity_15:output:0*M
_input_shapes<
:: ::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132(
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
?	
?
C__inference_dense2_layer_call_and_return_conditional_losses_9297273

inputs#
matmul_readvariableop_kernel_23"
biasadd_readvariableop_bias_23
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_kernel_23*
_output_shapes
:	 ?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_23*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?2
?
D__inference_model_4_layer_call_and_return_conditional_losses_9297418
	input_img
conv1_kernel_20
conv1_bias_20
conv2_kernel_21
conv2_bias_21
dense1_kernel_22
dense1_bias_22
dense2_kernel_23
dense2_bias_23
dec_conv1_kernel_24
dec_conv1_bias_24
dec_conv2_kernel_25
dec_conv2_bias_25
dec_conv3_kernel_26
dec_conv3_bias_26
identity??conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?!dec_conv1/StatefulPartitionedCall?!dec_conv2/StatefulPartitionedCall?!dec_conv3/StatefulPartitionedCall?dense1/StatefulPartitionedCall?dense2/StatefulPartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCall	input_imgconv1_kernel_20conv1_bias_20*
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
B__inference_conv1_layer_call_and_return_conditional_losses_92971902
conv1/StatefulPartitionedCall?
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0conv2_kernel_21conv2_bias_21*
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
B__inference_conv2_layer_call_and_return_conditional_losses_92972132
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
D__inference_flatten_layer_call_and_return_conditional_losses_92972312
flatten/PartitionedCall?
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_kernel_22dense1_bias_22*
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
C__inference_dense1_layer_call_and_return_conditional_losses_92972502 
dense1/StatefulPartitionedCall?
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_kernel_23dense2_bias_23*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_dense2_layer_call_and_return_conditional_losses_92972732 
dense2/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall'dense2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_92972992
reshape/PartitionedCall?
!dec_conv1/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0dec_conv1_kernel_24dec_conv1_bias_24*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dec_conv1_layer_call_and_return_conditional_losses_92973182#
!dec_conv1/StatefulPartitionedCall?
up_sampling2d/PartitionedCallPartitionedCall*dec_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_92971412
up_sampling2d/PartitionedCall?
!dec_conv2/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0dec_conv2_kernel_25dec_conv2_bias_25*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dec_conv2_layer_call_and_return_conditional_losses_92973472#
!dec_conv2/StatefulPartitionedCall?
up_sampling2d_1/PartitionedCallPartitionedCall*dec_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_92971722!
up_sampling2d_1/PartitionedCall?
!dec_conv3/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0dec_conv3_kernel_26dec_conv3_bias_26*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dec_conv3_layer_call_and_return_conditional_losses_92973762#
!dec_conv3/StatefulPartitionedCall?
IdentityIdentity*dec_conv3/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall"^dec_conv1/StatefulPartitionedCall"^dec_conv2/StatefulPartitionedCall"^dec_conv3/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????@@::::::::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2F
!dec_conv1/StatefulPartitionedCall!dec_conv1/StatefulPartitionedCall2F
!dec_conv2/StatefulPartitionedCall!dec_conv2/StatefulPartitionedCall2F
!dec_conv3/StatefulPartitionedCall!dec_conv3/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall:Z V
/
_output_shapes
:?????????@@
#
_user_specified_name	input_img
?	
?
C__inference_dense2_layer_call_and_return_conditional_losses_9297812

inputs#
matmul_readvariableop_kernel_23"
biasadd_readvariableop_bias_23
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_kernel_23*
_output_shapes
:	 ?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_23*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?h
?	
D__inference_model_4_layer_call_and_return_conditional_losses_9297698

inputs)
%conv1_conv2d_readvariableop_kernel_20(
$conv1_biasadd_readvariableop_bias_20)
%conv2_conv2d_readvariableop_kernel_21(
$conv2_biasadd_readvariableop_bias_21*
&dense1_matmul_readvariableop_kernel_22)
%dense1_biasadd_readvariableop_bias_22*
&dense2_matmul_readvariableop_kernel_23)
%dense2_biasadd_readvariableop_bias_23-
)dec_conv1_conv2d_readvariableop_kernel_24,
(dec_conv1_biasadd_readvariableop_bias_24-
)dec_conv2_conv2d_readvariableop_kernel_25,
(dec_conv2_biasadd_readvariableop_bias_25-
)dec_conv3_conv2d_readvariableop_kernel_26,
(dec_conv3_biasadd_readvariableop_bias_26
identity??conv1/BiasAdd/ReadVariableOp?conv1/Conv2D/ReadVariableOp?conv2/BiasAdd/ReadVariableOp?conv2/Conv2D/ReadVariableOp? dec_conv1/BiasAdd/ReadVariableOp?dec_conv1/Conv2D/ReadVariableOp? dec_conv2/BiasAdd/ReadVariableOp?dec_conv2/Conv2D/ReadVariableOp? dec_conv3/BiasAdd/ReadVariableOp?dec_conv3/Conv2D/ReadVariableOp?dense1/BiasAdd/ReadVariableOp?dense1/MatMul/ReadVariableOp?dense2/BiasAdd/ReadVariableOp?dense2/MatMul/ReadVariableOp?
conv1/Conv2D/ReadVariableOpReadVariableOp%conv1_conv2d_readvariableop_kernel_20*&
_output_shapes
:*
dtype02
conv1/Conv2D/ReadVariableOp?
conv1/Conv2DConv2Dinputs#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1/Conv2D?
conv1/BiasAdd/ReadVariableOpReadVariableOp$conv1_biasadd_readvariableop_bias_20*
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
conv2/Conv2D/ReadVariableOpReadVariableOp%conv2_conv2d_readvariableop_kernel_21*&
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
conv2/BiasAdd/ReadVariableOpReadVariableOp$conv2_biasadd_readvariableop_bias_21*
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
dense1/MatMul/ReadVariableOpReadVariableOp&dense1_matmul_readvariableop_kernel_22*
_output_shapes
:	? *
dtype02
dense1/MatMul/ReadVariableOp?
dense1/MatMulMatMulflatten/Reshape:output:0$dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense1/MatMul?
dense1/BiasAdd/ReadVariableOpReadVariableOp%dense1_biasadd_readvariableop_bias_22*
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
dense1/Relu?
dense2/MatMul/ReadVariableOpReadVariableOp&dense2_matmul_readvariableop_kernel_23*
_output_shapes
:	 ?*
dtype02
dense2/MatMul/ReadVariableOp?
dense2/MatMulMatMuldense1/Relu:activations:0$dense2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense2/MatMul?
dense2/BiasAdd/ReadVariableOpReadVariableOp%dense2_biasadd_readvariableop_bias_23*
_output_shapes	
:?*
dtype02
dense2/BiasAdd/ReadVariableOp?
dense2/BiasAddBiasAdddense2/MatMul:product:0%dense2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense2/BiasAddn
dense2/ReluReludense2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense2/Relug
reshape/ShapeShapedense2/Relu:activations:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/3?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapedense2/Relu:activations:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
reshape/Reshape?
dec_conv1/Conv2D/ReadVariableOpReadVariableOp)dec_conv1_conv2d_readvariableop_kernel_24*&
_output_shapes
:*
dtype02!
dec_conv1/Conv2D/ReadVariableOp?
dec_conv1/Conv2DConv2Dreshape/Reshape:output:0'dec_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
dec_conv1/Conv2D?
 dec_conv1/BiasAdd/ReadVariableOpReadVariableOp(dec_conv1_biasadd_readvariableop_bias_24*
_output_shapes
:*
dtype02"
 dec_conv1/BiasAdd/ReadVariableOp?
dec_conv1/BiasAddBiasAdddec_conv1/Conv2D:output:0(dec_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
dec_conv1/BiasAdd~
dec_conv1/ReluReludec_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
dec_conv1/Reluv
up_sampling2d/ShapeShapedec_conv1/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d/Shape?
!up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!up_sampling2d/strided_slice/stack?
#up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_1?
#up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_2?
up_sampling2d/strided_sliceStridedSliceup_sampling2d/Shape:output:0*up_sampling2d/strided_slice/stack:output:0,up_sampling2d/strided_slice/stack_1:output:0,up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d/strided_slice{
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const?
up_sampling2d/mulMul$up_sampling2d/strided_slice:output:0up_sampling2d/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d/mul?
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbordec_conv1/Relu:activations:0up_sampling2d/mul:z:0*
T0*/
_output_shapes
:?????????*
half_pixel_centers(2,
*up_sampling2d/resize/ResizeNearestNeighbor?
dec_conv2/Conv2D/ReadVariableOpReadVariableOp)dec_conv2_conv2d_readvariableop_kernel_25*&
_output_shapes
:
*
dtype02!
dec_conv2/Conv2D/ReadVariableOp?
dec_conv2/Conv2DConv2D;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0'dec_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
dec_conv2/Conv2D?
 dec_conv2/BiasAdd/ReadVariableOpReadVariableOp(dec_conv2_biasadd_readvariableop_bias_25*
_output_shapes
:*
dtype02"
 dec_conv2/BiasAdd/ReadVariableOp?
dec_conv2/BiasAddBiasAdddec_conv2/Conv2D:output:0(dec_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
dec_conv2/BiasAdd~
dec_conv2/ReluReludec_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
dec_conv2/Reluz
up_sampling2d_1/ShapeShapedec_conv2/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_1/Shape?
#up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_1/strided_slice/stack?
%up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_1?
%up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_2?
up_sampling2d_1/strided_sliceStridedSliceup_sampling2d_1/Shape:output:0,up_sampling2d_1/strided_slice/stack:output:0.up_sampling2d_1/strided_slice/stack_1:output:0.up_sampling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_1/strided_slice
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const?
up_sampling2d_1/mulMul&up_sampling2d_1/strided_slice:output:0up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/mul?
,up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbordec_conv2/Relu:activations:0up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:?????????KK*
half_pixel_centers(2.
,up_sampling2d_1/resize/ResizeNearestNeighbor?
dec_conv3/Conv2D/ReadVariableOpReadVariableOp)dec_conv3_conv2d_readvariableop_kernel_26*&
_output_shapes
:*
dtype02!
dec_conv3/Conv2D/ReadVariableOp?
dec_conv3/Conv2DConv2D=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0'dec_conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingVALID*
strides
2
dec_conv3/Conv2D?
 dec_conv3/BiasAdd/ReadVariableOpReadVariableOp(dec_conv3_biasadd_readvariableop_bias_26*
_output_shapes
:*
dtype02"
 dec_conv3/BiasAdd/ReadVariableOp?
dec_conv3/BiasAddBiasAdddec_conv3/Conv2D:output:0(dec_conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
dec_conv3/BiasAdd~
dec_conv3/ReluReludec_conv3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2
dec_conv3/Relu?
IdentityIdentitydec_conv3/Relu:activations:0^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp!^dec_conv1/BiasAdd/ReadVariableOp ^dec_conv1/Conv2D/ReadVariableOp!^dec_conv2/BiasAdd/ReadVariableOp ^dec_conv2/Conv2D/ReadVariableOp!^dec_conv3/BiasAdd/ReadVariableOp ^dec_conv3/Conv2D/ReadVariableOp^dense1/BiasAdd/ReadVariableOp^dense1/MatMul/ReadVariableOp^dense2/BiasAdd/ReadVariableOp^dense2/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????@@::::::::::::::2<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp2:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp2<
conv2/BiasAdd/ReadVariableOpconv2/BiasAdd/ReadVariableOp2:
conv2/Conv2D/ReadVariableOpconv2/Conv2D/ReadVariableOp2D
 dec_conv1/BiasAdd/ReadVariableOp dec_conv1/BiasAdd/ReadVariableOp2B
dec_conv1/Conv2D/ReadVariableOpdec_conv1/Conv2D/ReadVariableOp2D
 dec_conv2/BiasAdd/ReadVariableOp dec_conv2/BiasAdd/ReadVariableOp2B
dec_conv2/Conv2D/ReadVariableOpdec_conv2/Conv2D/ReadVariableOp2D
 dec_conv3/BiasAdd/ReadVariableOp dec_conv3/BiasAdd/ReadVariableOp2B
dec_conv3/Conv2D/ReadVariableOpdec_conv3/Conv2D/ReadVariableOp2>
dense1/BiasAdd/ReadVariableOpdense1/BiasAdd/ReadVariableOp2<
dense1/MatMul/ReadVariableOpdense1/MatMul/ReadVariableOp2>
dense2/BiasAdd/ReadVariableOpdense2/BiasAdd/ReadVariableOp2<
dense2/MatMul/ReadVariableOpdense2/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
}
(__inference_dense1_layer_call_fn_9297801

inputs
	kernel_22
bias_22
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs	kernel_22bias_22*
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
C__inference_dense1_layer_call_and_return_conditional_losses_92972502
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
B__inference_conv1_layer_call_and_return_conditional_losses_9297747

inputs#
conv2d_readvariableop_kernel_20"
biasadd_readvariableop_bias_20
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_kernel_20*&
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
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_20*
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
?

?
)__inference_model_4_layer_call_fn_9297467
	input_img
	kernel_20
bias_20
	kernel_21
bias_21
	kernel_22
bias_22
	kernel_23
bias_23
	kernel_24
bias_24
	kernel_25
bias_25
	kernel_26
bias_26
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	input_img	kernel_20bias_20	kernel_21bias_21	kernel_22bias_22	kernel_23bias_23	kernel_24bias_24	kernel_25bias_25	kernel_26bias_26*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_model_4_layer_call_and_return_conditional_losses_92974502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????@@::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
/
_output_shapes
:?????????@@
#
_user_specified_name	input_img
?
f
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_9297126

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_dec_conv2_layer_call_and_return_conditional_losses_9297867

inputs#
conv2d_readvariableop_kernel_25"
biasadd_readvariableop_bias_25
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_kernel_25*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_25*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
M
1__inference_up_sampling2d_1_layer_call_fn_9297175

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_92971722
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
`
D__inference_reshape_layer_call_and_return_conditional_losses_9297299

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
}
(__inference_dense2_layer_call_fn_9297819

inputs
	kernel_23
bias_23
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs	kernel_23bias_23*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_dense2_layer_call_and_return_conditional_losses_92972732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
h
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_9297172

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
F__inference_dec_conv1_layer_call_and_return_conditional_losses_9297318

inputs#
conv2d_readvariableop_kernel_24"
biasadd_readvariableop_bias_24
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_kernel_24*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_24*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_layer_call_and_return_conditional_losses_9297778

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
?
%__inference_signature_wrapper_9297536
	input_img
	kernel_20
bias_20
	kernel_21
bias_21
	kernel_22
bias_22
	kernel_23
bias_23
	kernel_24
bias_24
	kernel_25
bias_25
	kernel_26
bias_26
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	input_img	kernel_20bias_20	kernel_21bias_21	kernel_22bias_22	kernel_23bias_23	kernel_24bias_24	kernel_25bias_25	kernel_26bias_26*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8? *+
f&R$
"__inference__wrapped_model_92971132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????@@::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
/
_output_shapes
:?????????@@
#
_user_specified_name	input_img
?
?
+__inference_dec_conv2_layer_call_fn_9297874

inputs
	kernel_25
bias_25
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs	kernel_25bias_25*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dec_conv2_layer_call_and_return_conditional_losses_92973472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
C__inference_dense1_layer_call_and_return_conditional_losses_9297250

inputs#
matmul_readvariableop_kernel_22"
biasadd_readvariableop_bias_22
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_kernel_22*
_output_shapes
:	? *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_22*
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
B__inference_conv1_layer_call_and_return_conditional_losses_9297190

inputs#
conv2d_readvariableop_kernel_20"
biasadd_readvariableop_bias_20
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_kernel_20*&
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
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_20*
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
?

?
)__inference_model_4_layer_call_fn_9297736

inputs
	kernel_20
bias_20
	kernel_21
bias_21
	kernel_22
bias_22
	kernel_23
bias_23
	kernel_24
bias_24
	kernel_25
bias_25
	kernel_26
bias_26
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs	kernel_20bias_20	kernel_21bias_21	kernel_22bias_22	kernel_23bias_23	kernel_24bias_24	kernel_25bias_25	kernel_26bias_26*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_model_4_layer_call_and_return_conditional_losses_92974982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????@@::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?

?
)__inference_model_4_layer_call_fn_9297717

inputs
	kernel_20
bias_20
	kernel_21
bias_21
	kernel_22
bias_22
	kernel_23
bias_23
	kernel_24
bias_24
	kernel_25
bias_25
	kernel_26
bias_26
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs	kernel_20bias_20	kernel_21bias_21	kernel_22bias_22	kernel_23bias_23	kernel_24bias_24	kernel_25bias_25	kernel_26bias_26*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_model_4_layer_call_and_return_conditional_losses_92974502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????@@::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
+__inference_dec_conv1_layer_call_fn_9297856

inputs
	kernel_24
bias_24
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs	kernel_24bias_24*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dec_conv1_layer_call_and_return_conditional_losses_92973182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_reshape_layer_call_and_return_conditional_losses_9297833

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?2
?
D__inference_model_4_layer_call_and_return_conditional_losses_9297450

inputs
conv1_kernel_20
conv1_bias_20
conv2_kernel_21
conv2_bias_21
dense1_kernel_22
dense1_bias_22
dense2_kernel_23
dense2_bias_23
dec_conv1_kernel_24
dec_conv1_bias_24
dec_conv2_kernel_25
dec_conv2_bias_25
dec_conv3_kernel_26
dec_conv3_bias_26
identity??conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?!dec_conv1/StatefulPartitionedCall?!dec_conv2/StatefulPartitionedCall?!dec_conv3/StatefulPartitionedCall?dense1/StatefulPartitionedCall?dense2/StatefulPartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_kernel_20conv1_bias_20*
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
B__inference_conv1_layer_call_and_return_conditional_losses_92971902
conv1/StatefulPartitionedCall?
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0conv2_kernel_21conv2_bias_21*
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
B__inference_conv2_layer_call_and_return_conditional_losses_92972132
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
D__inference_flatten_layer_call_and_return_conditional_losses_92972312
flatten/PartitionedCall?
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_kernel_22dense1_bias_22*
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
C__inference_dense1_layer_call_and_return_conditional_losses_92972502 
dense1/StatefulPartitionedCall?
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_kernel_23dense2_bias_23*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *L
fGRE
C__inference_dense2_layer_call_and_return_conditional_losses_92972732 
dense2/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall'dense2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_92972992
reshape/PartitionedCall?
!dec_conv1/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0dec_conv1_kernel_24dec_conv1_bias_24*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dec_conv1_layer_call_and_return_conditional_losses_92973182#
!dec_conv1/StatefulPartitionedCall?
up_sampling2d/PartitionedCallPartitionedCall*dec_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *S
fNRL
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_92971412
up_sampling2d/PartitionedCall?
!dec_conv2/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0dec_conv2_kernel_25dec_conv2_bias_25*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dec_conv2_layer_call_and_return_conditional_losses_92973472#
!dec_conv2/StatefulPartitionedCall?
up_sampling2d_1/PartitionedCallPartitionedCall*dec_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8? *U
fPRN
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_92971722!
up_sampling2d_1/PartitionedCall?
!dec_conv3/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0dec_conv3_kernel_26dec_conv3_bias_26*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8? *O
fJRH
F__inference_dec_conv3_layer_call_and_return_conditional_losses_92973762#
!dec_conv3/StatefulPartitionedCall?
IdentityIdentity*dec_conv3/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall"^dec_conv1/StatefulPartitionedCall"^dec_conv2/StatefulPartitionedCall"^dec_conv3/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????@@::::::::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2F
!dec_conv1/StatefulPartitionedCall!dec_conv1/StatefulPartitionedCall2F
!dec_conv2/StatefulPartitionedCall!dec_conv2/StatefulPartitionedCall2F
!dec_conv3/StatefulPartitionedCall!dec_conv3/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?
?
F__inference_dec_conv3_layer_call_and_return_conditional_losses_9297376

inputs#
conv2d_readvariableop_kernel_26"
biasadd_readvariableop_bias_26
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_kernel_26*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_26*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?

?
F__inference_dec_conv1_layer_call_and_return_conditional_losses_9297849

inputs#
conv2d_readvariableop_kernel_24"
biasadd_readvariableop_bias_24
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_kernel_24*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_24*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_layer_call_and_return_conditional_losses_9297231

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
?
E
)__inference_flatten_layer_call_fn_9297783

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
D__inference_flatten_layer_call_and_return_conditional_losses_92972312
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
?

?
)__inference_model_4_layer_call_fn_9297515
	input_img
	kernel_20
bias_20
	kernel_21
bias_21
	kernel_22
bias_22
	kernel_23
bias_23
	kernel_24
bias_24
	kernel_25
bias_25
	kernel_26
bias_26
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	input_img	kernel_20bias_20	kernel_21bias_21	kernel_22bias_22	kernel_23bias_23	kernel_24bias_24	kernel_25bias_25	kernel_26bias_26*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*0
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_model_4_layer_call_and_return_conditional_losses_92974982
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????@@::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
/
_output_shapes
:?????????@@
#
_user_specified_name	input_img
?

?
B__inference_conv2_layer_call_and_return_conditional_losses_9297213

inputs#
conv2d_readvariableop_kernel_21"
biasadd_readvariableop_bias_21
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_kernel_21*&
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
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_21*
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
?y
?

"__inference__wrapped_model_9297113
	input_img1
-model_4_conv1_conv2d_readvariableop_kernel_200
,model_4_conv1_biasadd_readvariableop_bias_201
-model_4_conv2_conv2d_readvariableop_kernel_210
,model_4_conv2_biasadd_readvariableop_bias_212
.model_4_dense1_matmul_readvariableop_kernel_221
-model_4_dense1_biasadd_readvariableop_bias_222
.model_4_dense2_matmul_readvariableop_kernel_231
-model_4_dense2_biasadd_readvariableop_bias_235
1model_4_dec_conv1_conv2d_readvariableop_kernel_244
0model_4_dec_conv1_biasadd_readvariableop_bias_245
1model_4_dec_conv2_conv2d_readvariableop_kernel_254
0model_4_dec_conv2_biasadd_readvariableop_bias_255
1model_4_dec_conv3_conv2d_readvariableop_kernel_264
0model_4_dec_conv3_biasadd_readvariableop_bias_26
identity??$model_4/conv1/BiasAdd/ReadVariableOp?#model_4/conv1/Conv2D/ReadVariableOp?$model_4/conv2/BiasAdd/ReadVariableOp?#model_4/conv2/Conv2D/ReadVariableOp?(model_4/dec_conv1/BiasAdd/ReadVariableOp?'model_4/dec_conv1/Conv2D/ReadVariableOp?(model_4/dec_conv2/BiasAdd/ReadVariableOp?'model_4/dec_conv2/Conv2D/ReadVariableOp?(model_4/dec_conv3/BiasAdd/ReadVariableOp?'model_4/dec_conv3/Conv2D/ReadVariableOp?%model_4/dense1/BiasAdd/ReadVariableOp?$model_4/dense1/MatMul/ReadVariableOp?%model_4/dense2/BiasAdd/ReadVariableOp?$model_4/dense2/MatMul/ReadVariableOp?
#model_4/conv1/Conv2D/ReadVariableOpReadVariableOp-model_4_conv1_conv2d_readvariableop_kernel_20*&
_output_shapes
:*
dtype02%
#model_4/conv1/Conv2D/ReadVariableOp?
model_4/conv1/Conv2DConv2D	input_img+model_4/conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
model_4/conv1/Conv2D?
$model_4/conv1/BiasAdd/ReadVariableOpReadVariableOp,model_4_conv1_biasadd_readvariableop_bias_20*
_output_shapes
:*
dtype02&
$model_4/conv1/BiasAdd/ReadVariableOp?
model_4/conv1/BiasAddBiasAddmodel_4/conv1/Conv2D:output:0,model_4/conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
model_4/conv1/BiasAdd?
model_4/conv1/ReluRelumodel_4/conv1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
model_4/conv1/Relu?
#model_4/conv2/Conv2D/ReadVariableOpReadVariableOp-model_4_conv2_conv2d_readvariableop_kernel_21*&
_output_shapes
:*
dtype02%
#model_4/conv2/Conv2D/ReadVariableOp?
model_4/conv2/Conv2DConv2D model_4/conv1/Relu:activations:0+model_4/conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
model_4/conv2/Conv2D?
$model_4/conv2/BiasAdd/ReadVariableOpReadVariableOp,model_4_conv2_biasadd_readvariableop_bias_21*
_output_shapes
:*
dtype02&
$model_4/conv2/BiasAdd/ReadVariableOp?
model_4/conv2/BiasAddBiasAddmodel_4/conv2/Conv2D:output:0,model_4/conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
model_4/conv2/BiasAdd?
model_4/conv2/ReluRelumodel_4/conv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
model_4/conv2/Relu
model_4/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  2
model_4/flatten/Const?
model_4/flatten/ReshapeReshape model_4/conv2/Relu:activations:0model_4/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
model_4/flatten/Reshape?
$model_4/dense1/MatMul/ReadVariableOpReadVariableOp.model_4_dense1_matmul_readvariableop_kernel_22*
_output_shapes
:	? *
dtype02&
$model_4/dense1/MatMul/ReadVariableOp?
model_4/dense1/MatMulMatMul model_4/flatten/Reshape:output:0,model_4/dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
model_4/dense1/MatMul?
%model_4/dense1/BiasAdd/ReadVariableOpReadVariableOp-model_4_dense1_biasadd_readvariableop_bias_22*
_output_shapes
: *
dtype02'
%model_4/dense1/BiasAdd/ReadVariableOp?
model_4/dense1/BiasAddBiasAddmodel_4/dense1/MatMul:product:0-model_4/dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
model_4/dense1/BiasAdd?
model_4/dense1/ReluRelumodel_4/dense1/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
model_4/dense1/Relu?
$model_4/dense2/MatMul/ReadVariableOpReadVariableOp.model_4_dense2_matmul_readvariableop_kernel_23*
_output_shapes
:	 ?*
dtype02&
$model_4/dense2/MatMul/ReadVariableOp?
model_4/dense2/MatMulMatMul!model_4/dense1/Relu:activations:0,model_4/dense2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_4/dense2/MatMul?
%model_4/dense2/BiasAdd/ReadVariableOpReadVariableOp-model_4_dense2_biasadd_readvariableop_bias_23*
_output_shapes	
:?*
dtype02'
%model_4/dense2/BiasAdd/ReadVariableOp?
model_4/dense2/BiasAddBiasAddmodel_4/dense2/MatMul:product:0-model_4/dense2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_4/dense2/BiasAdd?
model_4/dense2/ReluRelumodel_4/dense2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model_4/dense2/Relu
model_4/reshape/ShapeShape!model_4/dense2/Relu:activations:0*
T0*
_output_shapes
:2
model_4/reshape/Shape?
#model_4/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model_4/reshape/strided_slice/stack?
%model_4/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%model_4/reshape/strided_slice/stack_1?
%model_4/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%model_4/reshape/strided_slice/stack_2?
model_4/reshape/strided_sliceStridedSlicemodel_4/reshape/Shape:output:0,model_4/reshape/strided_slice/stack:output:0.model_4/reshape/strided_slice/stack_1:output:0.model_4/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model_4/reshape/strided_slice?
model_4/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2!
model_4/reshape/Reshape/shape/1?
model_4/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2!
model_4/reshape/Reshape/shape/2?
model_4/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2!
model_4/reshape/Reshape/shape/3?
model_4/reshape/Reshape/shapePack&model_4/reshape/strided_slice:output:0(model_4/reshape/Reshape/shape/1:output:0(model_4/reshape/Reshape/shape/2:output:0(model_4/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
model_4/reshape/Reshape/shape?
model_4/reshape/ReshapeReshape!model_4/dense2/Relu:activations:0&model_4/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
model_4/reshape/Reshape?
'model_4/dec_conv1/Conv2D/ReadVariableOpReadVariableOp1model_4_dec_conv1_conv2d_readvariableop_kernel_24*&
_output_shapes
:*
dtype02)
'model_4/dec_conv1/Conv2D/ReadVariableOp?
model_4/dec_conv1/Conv2DConv2D model_4/reshape/Reshape:output:0/model_4/dec_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
model_4/dec_conv1/Conv2D?
(model_4/dec_conv1/BiasAdd/ReadVariableOpReadVariableOp0model_4_dec_conv1_biasadd_readvariableop_bias_24*
_output_shapes
:*
dtype02*
(model_4/dec_conv1/BiasAdd/ReadVariableOp?
model_4/dec_conv1/BiasAddBiasAdd!model_4/dec_conv1/Conv2D:output:00model_4/dec_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
model_4/dec_conv1/BiasAdd?
model_4/dec_conv1/ReluRelu"model_4/dec_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
model_4/dec_conv1/Relu?
model_4/up_sampling2d/ShapeShape$model_4/dec_conv1/Relu:activations:0*
T0*
_output_shapes
:2
model_4/up_sampling2d/Shape?
)model_4/up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)model_4/up_sampling2d/strided_slice/stack?
+model_4/up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+model_4/up_sampling2d/strided_slice/stack_1?
+model_4/up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+model_4/up_sampling2d/strided_slice/stack_2?
#model_4/up_sampling2d/strided_sliceStridedSlice$model_4/up_sampling2d/Shape:output:02model_4/up_sampling2d/strided_slice/stack:output:04model_4/up_sampling2d/strided_slice/stack_1:output:04model_4/up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2%
#model_4/up_sampling2d/strided_slice?
model_4/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
model_4/up_sampling2d/Const?
model_4/up_sampling2d/mulMul,model_4/up_sampling2d/strided_slice:output:0$model_4/up_sampling2d/Const:output:0*
T0*
_output_shapes
:2
model_4/up_sampling2d/mul?
2model_4/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor$model_4/dec_conv1/Relu:activations:0model_4/up_sampling2d/mul:z:0*
T0*/
_output_shapes
:?????????*
half_pixel_centers(24
2model_4/up_sampling2d/resize/ResizeNearestNeighbor?
'model_4/dec_conv2/Conv2D/ReadVariableOpReadVariableOp1model_4_dec_conv2_conv2d_readvariableop_kernel_25*&
_output_shapes
:
*
dtype02)
'model_4/dec_conv2/Conv2D/ReadVariableOp?
model_4/dec_conv2/Conv2DConv2DCmodel_4/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0/model_4/dec_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
model_4/dec_conv2/Conv2D?
(model_4/dec_conv2/BiasAdd/ReadVariableOpReadVariableOp0model_4_dec_conv2_biasadd_readvariableop_bias_25*
_output_shapes
:*
dtype02*
(model_4/dec_conv2/BiasAdd/ReadVariableOp?
model_4/dec_conv2/BiasAddBiasAdd!model_4/dec_conv2/Conv2D:output:00model_4/dec_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
model_4/dec_conv2/BiasAdd?
model_4/dec_conv2/ReluRelu"model_4/dec_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
model_4/dec_conv2/Relu?
model_4/up_sampling2d_1/ShapeShape$model_4/dec_conv2/Relu:activations:0*
T0*
_output_shapes
:2
model_4/up_sampling2d_1/Shape?
+model_4/up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+model_4/up_sampling2d_1/strided_slice/stack?
-model_4/up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-model_4/up_sampling2d_1/strided_slice/stack_1?
-model_4/up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-model_4/up_sampling2d_1/strided_slice/stack_2?
%model_4/up_sampling2d_1/strided_sliceStridedSlice&model_4/up_sampling2d_1/Shape:output:04model_4/up_sampling2d_1/strided_slice/stack:output:06model_4/up_sampling2d_1/strided_slice/stack_1:output:06model_4/up_sampling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2'
%model_4/up_sampling2d_1/strided_slice?
model_4/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
model_4/up_sampling2d_1/Const?
model_4/up_sampling2d_1/mulMul.model_4/up_sampling2d_1/strided_slice:output:0&model_4/up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2
model_4/up_sampling2d_1/mul?
4model_4/up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor$model_4/dec_conv2/Relu:activations:0model_4/up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:?????????KK*
half_pixel_centers(26
4model_4/up_sampling2d_1/resize/ResizeNearestNeighbor?
'model_4/dec_conv3/Conv2D/ReadVariableOpReadVariableOp1model_4_dec_conv3_conv2d_readvariableop_kernel_26*&
_output_shapes
:*
dtype02)
'model_4/dec_conv3/Conv2D/ReadVariableOp?
model_4/dec_conv3/Conv2DConv2DEmodel_4/up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0/model_4/dec_conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingVALID*
strides
2
model_4/dec_conv3/Conv2D?
(model_4/dec_conv3/BiasAdd/ReadVariableOpReadVariableOp0model_4_dec_conv3_biasadd_readvariableop_bias_26*
_output_shapes
:*
dtype02*
(model_4/dec_conv3/BiasAdd/ReadVariableOp?
model_4/dec_conv3/BiasAddBiasAdd!model_4/dec_conv3/Conv2D:output:00model_4/dec_conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
model_4/dec_conv3/BiasAdd?
model_4/dec_conv3/ReluRelu"model_4/dec_conv3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2
model_4/dec_conv3/Relu?
IdentityIdentity$model_4/dec_conv3/Relu:activations:0%^model_4/conv1/BiasAdd/ReadVariableOp$^model_4/conv1/Conv2D/ReadVariableOp%^model_4/conv2/BiasAdd/ReadVariableOp$^model_4/conv2/Conv2D/ReadVariableOp)^model_4/dec_conv1/BiasAdd/ReadVariableOp(^model_4/dec_conv1/Conv2D/ReadVariableOp)^model_4/dec_conv2/BiasAdd/ReadVariableOp(^model_4/dec_conv2/Conv2D/ReadVariableOp)^model_4/dec_conv3/BiasAdd/ReadVariableOp(^model_4/dec_conv3/Conv2D/ReadVariableOp&^model_4/dense1/BiasAdd/ReadVariableOp%^model_4/dense1/MatMul/ReadVariableOp&^model_4/dense2/BiasAdd/ReadVariableOp%^model_4/dense2/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????@@::::::::::::::2L
$model_4/conv1/BiasAdd/ReadVariableOp$model_4/conv1/BiasAdd/ReadVariableOp2J
#model_4/conv1/Conv2D/ReadVariableOp#model_4/conv1/Conv2D/ReadVariableOp2L
$model_4/conv2/BiasAdd/ReadVariableOp$model_4/conv2/BiasAdd/ReadVariableOp2J
#model_4/conv2/Conv2D/ReadVariableOp#model_4/conv2/Conv2D/ReadVariableOp2T
(model_4/dec_conv1/BiasAdd/ReadVariableOp(model_4/dec_conv1/BiasAdd/ReadVariableOp2R
'model_4/dec_conv1/Conv2D/ReadVariableOp'model_4/dec_conv1/Conv2D/ReadVariableOp2T
(model_4/dec_conv2/BiasAdd/ReadVariableOp(model_4/dec_conv2/BiasAdd/ReadVariableOp2R
'model_4/dec_conv2/Conv2D/ReadVariableOp'model_4/dec_conv2/Conv2D/ReadVariableOp2T
(model_4/dec_conv3/BiasAdd/ReadVariableOp(model_4/dec_conv3/BiasAdd/ReadVariableOp2R
'model_4/dec_conv3/Conv2D/ReadVariableOp'model_4/dec_conv3/Conv2D/ReadVariableOp2N
%model_4/dense1/BiasAdd/ReadVariableOp%model_4/dense1/BiasAdd/ReadVariableOp2L
$model_4/dense1/MatMul/ReadVariableOp$model_4/dense1/MatMul/ReadVariableOp2N
%model_4/dense2/BiasAdd/ReadVariableOp%model_4/dense2/BiasAdd/ReadVariableOp2L
$model_4/dense2/MatMul/ReadVariableOp$model_4/dense2/MatMul/ReadVariableOp:Z V
/
_output_shapes
:?????????@@
#
_user_specified_name	input_img
?
f
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_9297141

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
h
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_9297157

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
|
'__inference_conv2_layer_call_fn_9297772

inputs
	kernel_21
bias_21
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs	kernel_21bias_21*
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
B__inference_conv2_layer_call_and_return_conditional_losses_92972132
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
?(
?
 __inference__traced_save_9297957
file_prefix+
'savev2_conv1_kernel_read_readvariableop)
%savev2_conv1_bias_read_readvariableop+
'savev2_conv2_kernel_read_readvariableop)
%savev2_conv2_bias_read_readvariableop,
(savev2_dense1_kernel_read_readvariableop*
&savev2_dense1_bias_read_readvariableop,
(savev2_dense2_kernel_read_readvariableop*
&savev2_dense2_bias_read_readvariableop/
+savev2_dec_conv1_kernel_read_readvariableop-
)savev2_dec_conv1_bias_read_readvariableop/
+savev2_dec_conv2_kernel_read_readvariableop-
)savev2_dec_conv2_bias_read_readvariableop/
+savev2_dec_conv3_kernel_read_readvariableop-
)savev2_dec_conv3_bias_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_conv1_kernel_read_readvariableop%savev2_conv1_bias_read_readvariableop'savev2_conv2_kernel_read_readvariableop%savev2_conv2_bias_read_readvariableop(savev2_dense1_kernel_read_readvariableop&savev2_dense1_bias_read_readvariableop(savev2_dense2_kernel_read_readvariableop&savev2_dense2_bias_read_readvariableop+savev2_dec_conv1_kernel_read_readvariableop)savev2_dec_conv1_bias_read_readvariableop+savev2_dec_conv2_kernel_read_readvariableop)savev2_dec_conv2_bias_read_readvariableop+savev2_dec_conv3_kernel_read_readvariableop)savev2_dec_conv3_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
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
?: :::::	? : :	 ?:?:::
:::: 2(
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
: :%!

_output_shapes
:	 ?:!

_output_shapes	
:?:,	(
&
_output_shapes
:: 


_output_shapes
::,(
&
_output_shapes
:
: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
?
?
F__inference_dec_conv2_layer_call_and_return_conditional_losses_9297347

inputs#
conv2d_readvariableop_kernel_25"
biasadd_readvariableop_bias_25
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_kernel_25*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_25*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?h
?	
D__inference_model_4_layer_call_and_return_conditional_losses_9297617

inputs)
%conv1_conv2d_readvariableop_kernel_20(
$conv1_biasadd_readvariableop_bias_20)
%conv2_conv2d_readvariableop_kernel_21(
$conv2_biasadd_readvariableop_bias_21*
&dense1_matmul_readvariableop_kernel_22)
%dense1_biasadd_readvariableop_bias_22*
&dense2_matmul_readvariableop_kernel_23)
%dense2_biasadd_readvariableop_bias_23-
)dec_conv1_conv2d_readvariableop_kernel_24,
(dec_conv1_biasadd_readvariableop_bias_24-
)dec_conv2_conv2d_readvariableop_kernel_25,
(dec_conv2_biasadd_readvariableop_bias_25-
)dec_conv3_conv2d_readvariableop_kernel_26,
(dec_conv3_biasadd_readvariableop_bias_26
identity??conv1/BiasAdd/ReadVariableOp?conv1/Conv2D/ReadVariableOp?conv2/BiasAdd/ReadVariableOp?conv2/Conv2D/ReadVariableOp? dec_conv1/BiasAdd/ReadVariableOp?dec_conv1/Conv2D/ReadVariableOp? dec_conv2/BiasAdd/ReadVariableOp?dec_conv2/Conv2D/ReadVariableOp? dec_conv3/BiasAdd/ReadVariableOp?dec_conv3/Conv2D/ReadVariableOp?dense1/BiasAdd/ReadVariableOp?dense1/MatMul/ReadVariableOp?dense2/BiasAdd/ReadVariableOp?dense2/MatMul/ReadVariableOp?
conv1/Conv2D/ReadVariableOpReadVariableOp%conv1_conv2d_readvariableop_kernel_20*&
_output_shapes
:*
dtype02
conv1/Conv2D/ReadVariableOp?
conv1/Conv2DConv2Dinputs#conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv1/Conv2D?
conv1/BiasAdd/ReadVariableOpReadVariableOp$conv1_biasadd_readvariableop_bias_20*
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
conv2/Conv2D/ReadVariableOpReadVariableOp%conv2_conv2d_readvariableop_kernel_21*&
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
conv2/BiasAdd/ReadVariableOpReadVariableOp$conv2_biasadd_readvariableop_bias_21*
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
dense1/MatMul/ReadVariableOpReadVariableOp&dense1_matmul_readvariableop_kernel_22*
_output_shapes
:	? *
dtype02
dense1/MatMul/ReadVariableOp?
dense1/MatMulMatMulflatten/Reshape:output:0$dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense1/MatMul?
dense1/BiasAdd/ReadVariableOpReadVariableOp%dense1_biasadd_readvariableop_bias_22*
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
dense1/Relu?
dense2/MatMul/ReadVariableOpReadVariableOp&dense2_matmul_readvariableop_kernel_23*
_output_shapes
:	 ?*
dtype02
dense2/MatMul/ReadVariableOp?
dense2/MatMulMatMuldense1/Relu:activations:0$dense2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense2/MatMul?
dense2/BiasAdd/ReadVariableOpReadVariableOp%dense2_biasadd_readvariableop_bias_23*
_output_shapes	
:?*
dtype02
dense2/BiasAdd/ReadVariableOp?
dense2/BiasAddBiasAdddense2/MatMul:product:0%dense2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense2/BiasAddn
dense2/ReluReludense2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense2/Relug
reshape/ShapeShapedense2/Relu:activations:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/3?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapedense2/Relu:activations:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
reshape/Reshape?
dec_conv1/Conv2D/ReadVariableOpReadVariableOp)dec_conv1_conv2d_readvariableop_kernel_24*&
_output_shapes
:*
dtype02!
dec_conv1/Conv2D/ReadVariableOp?
dec_conv1/Conv2DConv2Dreshape/Reshape:output:0'dec_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
dec_conv1/Conv2D?
 dec_conv1/BiasAdd/ReadVariableOpReadVariableOp(dec_conv1_biasadd_readvariableop_bias_24*
_output_shapes
:*
dtype02"
 dec_conv1/BiasAdd/ReadVariableOp?
dec_conv1/BiasAddBiasAdddec_conv1/Conv2D:output:0(dec_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
dec_conv1/BiasAdd~
dec_conv1/ReluReludec_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
dec_conv1/Reluv
up_sampling2d/ShapeShapedec_conv1/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d/Shape?
!up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!up_sampling2d/strided_slice/stack?
#up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_1?
#up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_2?
up_sampling2d/strided_sliceStridedSliceup_sampling2d/Shape:output:0*up_sampling2d/strided_slice/stack:output:0,up_sampling2d/strided_slice/stack_1:output:0,up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d/strided_slice{
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const?
up_sampling2d/mulMul$up_sampling2d/strided_slice:output:0up_sampling2d/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d/mul?
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbordec_conv1/Relu:activations:0up_sampling2d/mul:z:0*
T0*/
_output_shapes
:?????????*
half_pixel_centers(2,
*up_sampling2d/resize/ResizeNearestNeighbor?
dec_conv2/Conv2D/ReadVariableOpReadVariableOp)dec_conv2_conv2d_readvariableop_kernel_25*&
_output_shapes
:
*
dtype02!
dec_conv2/Conv2D/ReadVariableOp?
dec_conv2/Conv2DConv2D;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0'dec_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
dec_conv2/Conv2D?
 dec_conv2/BiasAdd/ReadVariableOpReadVariableOp(dec_conv2_biasadd_readvariableop_bias_25*
_output_shapes
:*
dtype02"
 dec_conv2/BiasAdd/ReadVariableOp?
dec_conv2/BiasAddBiasAdddec_conv2/Conv2D:output:0(dec_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
dec_conv2/BiasAdd~
dec_conv2/ReluReludec_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
dec_conv2/Reluz
up_sampling2d_1/ShapeShapedec_conv2/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_1/Shape?
#up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_1/strided_slice/stack?
%up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_1?
%up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_2?
up_sampling2d_1/strided_sliceStridedSliceup_sampling2d_1/Shape:output:0,up_sampling2d_1/strided_slice/stack:output:0.up_sampling2d_1/strided_slice/stack_1:output:0.up_sampling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_1/strided_slice
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const?
up_sampling2d_1/mulMul&up_sampling2d_1/strided_slice:output:0up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/mul?
,up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbordec_conv2/Relu:activations:0up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:?????????KK*
half_pixel_centers(2.
,up_sampling2d_1/resize/ResizeNearestNeighbor?
dec_conv3/Conv2D/ReadVariableOpReadVariableOp)dec_conv3_conv2d_readvariableop_kernel_26*&
_output_shapes
:*
dtype02!
dec_conv3/Conv2D/ReadVariableOp?
dec_conv3/Conv2DConv2D=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0'dec_conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingVALID*
strides
2
dec_conv3/Conv2D?
 dec_conv3/BiasAdd/ReadVariableOpReadVariableOp(dec_conv3_biasadd_readvariableop_bias_26*
_output_shapes
:*
dtype02"
 dec_conv3/BiasAdd/ReadVariableOp?
dec_conv3/BiasAddBiasAdddec_conv3/Conv2D:output:0(dec_conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
dec_conv3/BiasAdd~
dec_conv3/ReluReludec_conv3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2
dec_conv3/Relu?
IdentityIdentitydec_conv3/Relu:activations:0^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp!^dec_conv1/BiasAdd/ReadVariableOp ^dec_conv1/Conv2D/ReadVariableOp!^dec_conv2/BiasAdd/ReadVariableOp ^dec_conv2/Conv2D/ReadVariableOp!^dec_conv3/BiasAdd/ReadVariableOp ^dec_conv3/Conv2D/ReadVariableOp^dense1/BiasAdd/ReadVariableOp^dense1/MatMul/ReadVariableOp^dense2/BiasAdd/ReadVariableOp^dense2/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????@@2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:?????????@@::::::::::::::2<
conv1/BiasAdd/ReadVariableOpconv1/BiasAdd/ReadVariableOp2:
conv1/Conv2D/ReadVariableOpconv1/Conv2D/ReadVariableOp2<
conv2/BiasAdd/ReadVariableOpconv2/BiasAdd/ReadVariableOp2:
conv2/Conv2D/ReadVariableOpconv2/Conv2D/ReadVariableOp2D
 dec_conv1/BiasAdd/ReadVariableOp dec_conv1/BiasAdd/ReadVariableOp2B
dec_conv1/Conv2D/ReadVariableOpdec_conv1/Conv2D/ReadVariableOp2D
 dec_conv2/BiasAdd/ReadVariableOp dec_conv2/BiasAdd/ReadVariableOp2B
dec_conv2/Conv2D/ReadVariableOpdec_conv2/Conv2D/ReadVariableOp2D
 dec_conv3/BiasAdd/ReadVariableOp dec_conv3/BiasAdd/ReadVariableOp2B
dec_conv3/Conv2D/ReadVariableOpdec_conv3/Conv2D/ReadVariableOp2>
dense1/BiasAdd/ReadVariableOpdense1/BiasAdd/ReadVariableOp2<
dense1/MatMul/ReadVariableOpdense1/MatMul/ReadVariableOp2>
dense2/BiasAdd/ReadVariableOpdense2/BiasAdd/ReadVariableOp2<
dense2/MatMul/ReadVariableOpdense2/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????@@
 
_user_specified_nameinputs
?

?
B__inference_conv2_layer_call_and_return_conditional_losses_9297765

inputs#
conv2d_readvariableop_kernel_21"
biasadd_readvariableop_bias_21
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_kernel_21*&
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
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_21*
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
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
G
	input_img:
serving_default_input_img:0?????????@@E
	dec_conv38
StatefulPartitionedCall:0?????????@@tensorflow/serving/predict:??
?g
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11

signatures
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"?c
_tf_keras_network?b{"class_name": "Functional", "name": "model_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_img"}, "name": "input_img", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [8, 8]}, "strides": {"class_name": "__tuple__", "items": [4, 4]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1", "inbound_nodes": [[["input_img", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2", "inbound_nodes": [[["conv1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["conv2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense1", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense2", "trainable": true, "dtype": "float32", "units": 1536, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense2", "inbound_nodes": [[["dense1", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [12, 8, 16]}}, "name": "reshape", "inbound_nodes": [[["dense2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "dec_conv1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_conv1", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d", "inbound_nodes": [[["dec_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "dec_conv2", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [10, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_conv2", "inbound_nodes": [[["up_sampling2d", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [5, 5]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_1", "inbound_nodes": [[["dec_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "dec_conv3", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [12, 12]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_conv3", "inbound_nodes": [[["up_sampling2d_1", 0, 0, {}]]]}], "input_layers": [["input_img", 0, 0]], "output_layers": [["dec_conv3", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_4", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_img"}, "name": "input_img", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [8, 8]}, "strides": {"class_name": "__tuple__", "items": [4, 4]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1", "inbound_nodes": [[["input_img", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2", "inbound_nodes": [[["conv1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["conv2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense1", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense2", "trainable": true, "dtype": "float32", "units": 1536, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense2", "inbound_nodes": [[["dense1", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [12, 8, 16]}}, "name": "reshape", "inbound_nodes": [[["dense2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "dec_conv1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_conv1", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d", "inbound_nodes": [[["dec_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "dec_conv2", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [10, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_conv2", "inbound_nodes": [[["up_sampling2d", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [5, 5]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_1", "inbound_nodes": [[["dec_conv2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "dec_conv3", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [12, 12]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_conv3", "inbound_nodes": [[["up_sampling2d_1", 0, 0, {}]]]}], "input_layers": [["input_img", 0, 0]], "output_layers": [["dec_conv3", 0, 0]]}}}
?
#_self_saveable_object_factories"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_img", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_img"}}
?


kernel
bias
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [8, 8]}, "strides": {"class_name": "__tuple__", "items": [4, 4]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 3]}}
?


kernel
bias
#_self_saveable_object_factories
regularization_losses
	variables
 trainable_variables
!	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 15, 15, 8]}}
?
#"_self_saveable_object_factories
#regularization_losses
$	variables
%trainable_variables
&	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

'kernel
(bias
#)_self_saveable_object_factories
*regularization_losses
+	variables
,trainable_variables
-	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2304}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2304]}}
?

.kernel
/bias
#0_self_saveable_object_factories
1regularization_losses
2	variables
3trainable_variables
4	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense2", "trainable": true, "dtype": "float32", "units": 1536, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
?
#5_self_saveable_object_factories
6regularization_losses
7	variables
8trainable_variables
9	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [12, 8, 16]}}}
?


:kernel
;bias
#<_self_saveable_object_factories
=regularization_losses
>	variables
?trainable_variables
@	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "dec_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dec_conv1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 8, 16]}}
?
#A_self_saveable_object_factories
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "UpSampling2D", "name": "up_sampling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?


Fkernel
Gbias
#H_self_saveable_object_factories
Iregularization_losses
J	variables
Ktrainable_variables
L	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "dec_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dec_conv2", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [10, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 16, 16]}}
?
#M_self_saveable_object_factories
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "UpSampling2D", "name": "up_sampling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [5, 5]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?


Rkernel
Sbias
#T_self_saveable_object_factories
Uregularization_losses
V	variables
Wtrainable_variables
X	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "dec_conv3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dec_conv3", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [12, 12]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 75, 75, 8]}}
-
?serving_default"
signature_map
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?
0
1
2
3
'4
(5
.6
/7
:8
;9
F10
G11
R12
S13"
trackable_list_wrapper
?
0
1
2
3
'4
(5
.6
/7
:8
;9
F10
G11
R12
S13"
trackable_list_wrapper
?
Ylayer_metrics
regularization_losses
Zlayer_regularization_losses
[non_trainable_variables

\layers
	variables
]metrics
trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
^layer_metrics
_layer_regularization_losses
regularization_losses
`non_trainable_variables

alayers
	variables
bmetrics
trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$2conv2/kernel
:2
conv2/bias
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
clayer_metrics
dlayer_regularization_losses
regularization_losses
enon_trainable_variables

flayers
	variables
gmetrics
 trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
hlayer_metrics
ilayer_regularization_losses
#regularization_losses
jnon_trainable_variables

klayers
$	variables
lmetrics
%trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :	? 2dense1/kernel
: 2dense1/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
?
mlayer_metrics
nlayer_regularization_losses
*regularization_losses
onon_trainable_variables

players
+	variables
qmetrics
,trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :	 ?2dense2/kernel
:?2dense2/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
?
rlayer_metrics
slayer_regularization_losses
1regularization_losses
tnon_trainable_variables

ulayers
2	variables
vmetrics
3trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
wlayer_metrics
xlayer_regularization_losses
6regularization_losses
ynon_trainable_variables

zlayers
7	variables
{metrics
8trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(2dec_conv1/kernel
:2dec_conv1/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
?
|layer_metrics
}layer_regularization_losses
=regularization_losses
~non_trainable_variables

layers
>	variables
?metrics
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?layer_metrics
 ?layer_regularization_losses
Bregularization_losses
?non_trainable_variables
?layers
C	variables
?metrics
Dtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(
2dec_conv2/kernel
:2dec_conv2/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
Iregularization_losses
?non_trainable_variables
?layers
J	variables
?metrics
Ktrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
?layer_metrics
 ?layer_regularization_losses
Nregularization_losses
?non_trainable_variables
?layers
O	variables
?metrics
Ptrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(2dec_conv3/kernel
:2dec_conv3/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
Uregularization_losses
?non_trainable_variables
?layers
V	variables
?metrics
Wtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
v
0
1
2
3
4
5
6
7
	8

9
10
11"
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
)__inference_model_4_layer_call_fn_9297515
)__inference_model_4_layer_call_fn_9297717
)__inference_model_4_layer_call_fn_9297736
)__inference_model_4_layer_call_fn_9297467?
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
D__inference_model_4_layer_call_and_return_conditional_losses_9297418
D__inference_model_4_layer_call_and_return_conditional_losses_9297698
D__inference_model_4_layer_call_and_return_conditional_losses_9297617
D__inference_model_4_layer_call_and_return_conditional_losses_9297389?
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
"__inference__wrapped_model_9297113?
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
annotations? *0?-
+?(
	input_img?????????@@
?2?
'__inference_conv1_layer_call_fn_9297754?
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
B__inference_conv1_layer_call_and_return_conditional_losses_9297747?
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
'__inference_conv2_layer_call_fn_9297772?
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
B__inference_conv2_layer_call_and_return_conditional_losses_9297765?
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
)__inference_flatten_layer_call_fn_9297783?
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
D__inference_flatten_layer_call_and_return_conditional_losses_9297778?
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
(__inference_dense1_layer_call_fn_9297801?
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
C__inference_dense1_layer_call_and_return_conditional_losses_9297794?
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
(__inference_dense2_layer_call_fn_9297819?
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
C__inference_dense2_layer_call_and_return_conditional_losses_9297812?
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
)__inference_reshape_layer_call_fn_9297838?
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
D__inference_reshape_layer_call_and_return_conditional_losses_9297833?
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
+__inference_dec_conv1_layer_call_fn_9297856?
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
F__inference_dec_conv1_layer_call_and_return_conditional_losses_9297849?
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
?2?
/__inference_up_sampling2d_layer_call_fn_9297144?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_9297126?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
+__inference_dec_conv2_layer_call_fn_9297874?
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
F__inference_dec_conv2_layer_call_and_return_conditional_losses_9297867?
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
?2?
1__inference_up_sampling2d_1_layer_call_fn_9297175?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_9297157?
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
annotations? *@?=
;?84????????????????????????????????????
?2?
+__inference_dec_conv3_layer_call_fn_9297892?
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
F__inference_dec_conv3_layer_call_and_return_conditional_losses_9297885?
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
%__inference_signature_wrapper_9297536	input_img"?
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
"__inference__wrapped_model_9297113?'(./:;FGRS:?7
0?-
+?(
	input_img?????????@@
? "=?:
8
	dec_conv3+?(
	dec_conv3?????????@@?
B__inference_conv1_layer_call_and_return_conditional_losses_9297747l7?4
-?*
(?%
inputs?????????@@
? "-?*
#? 
0?????????
? ?
'__inference_conv1_layer_call_fn_9297754_7?4
-?*
(?%
inputs?????????@@
? " ???????????
B__inference_conv2_layer_call_and_return_conditional_losses_9297765l7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
'__inference_conv2_layer_call_fn_9297772_7?4
-?*
(?%
inputs?????????
? " ???????????
F__inference_dec_conv1_layer_call_and_return_conditional_losses_9297849l:;7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
+__inference_dec_conv1_layer_call_fn_9297856_:;7?4
-?*
(?%
inputs?????????
? " ???????????
F__inference_dec_conv2_layer_call_and_return_conditional_losses_9297867?FGI?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
+__inference_dec_conv2_layer_call_fn_9297874?FGI?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
F__inference_dec_conv3_layer_call_and_return_conditional_losses_9297885?RSI?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
+__inference_dec_conv3_layer_call_fn_9297892?RSI?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
C__inference_dense1_layer_call_and_return_conditional_losses_9297794]'(0?-
&?#
!?
inputs??????????
? "%?"
?
0????????? 
? |
(__inference_dense1_layer_call_fn_9297801P'(0?-
&?#
!?
inputs??????????
? "?????????? ?
C__inference_dense2_layer_call_and_return_conditional_losses_9297812].//?,
%?"
 ?
inputs????????? 
? "&?#
?
0??????????
? |
(__inference_dense2_layer_call_fn_9297819P.//?,
%?"
 ?
inputs????????? 
? "????????????
D__inference_flatten_layer_call_and_return_conditional_losses_9297778a7?4
-?*
(?%
inputs?????????
? "&?#
?
0??????????
? ?
)__inference_flatten_layer_call_fn_9297783T7?4
-?*
(?%
inputs?????????
? "????????????
D__inference_model_4_layer_call_and_return_conditional_losses_9297389?'(./:;FGRSB??
8?5
+?(
	input_img?????????@@
p

 
? "??<
5?2
0+???????????????????????????
? ?
D__inference_model_4_layer_call_and_return_conditional_losses_9297418?'(./:;FGRSB??
8?5
+?(
	input_img?????????@@
p 

 
? "??<
5?2
0+???????????????????????????
? ?
D__inference_model_4_layer_call_and_return_conditional_losses_9297617?'(./:;FGRS??<
5?2
(?%
inputs?????????@@
p

 
? "-?*
#? 
0?????????@@
? ?
D__inference_model_4_layer_call_and_return_conditional_losses_9297698?'(./:;FGRS??<
5?2
(?%
inputs?????????@@
p 

 
? "-?*
#? 
0?????????@@
? ?
)__inference_model_4_layer_call_fn_9297467?'(./:;FGRSB??
8?5
+?(
	input_img?????????@@
p

 
? "2?/+????????????????????????????
)__inference_model_4_layer_call_fn_9297515?'(./:;FGRSB??
8?5
+?(
	input_img?????????@@
p 

 
? "2?/+????????????????????????????
)__inference_model_4_layer_call_fn_9297717?'(./:;FGRS??<
5?2
(?%
inputs?????????@@
p

 
? "2?/+????????????????????????????
)__inference_model_4_layer_call_fn_9297736?'(./:;FGRS??<
5?2
(?%
inputs?????????@@
p 

 
? "2?/+????????????????????????????
D__inference_reshape_layer_call_and_return_conditional_losses_9297833a0?-
&?#
!?
inputs??????????
? "-?*
#? 
0?????????
? ?
)__inference_reshape_layer_call_fn_9297838T0?-
&?#
!?
inputs??????????
? " ???????????
%__inference_signature_wrapper_9297536?'(./:;FGRSG?D
? 
=?:
8
	input_img+?(
	input_img?????????@@"=?:
8
	dec_conv3+?(
	dec_conv3?????????@@?
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_9297157?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
1__inference_up_sampling2d_1_layer_call_fn_9297175?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_9297126?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_up_sampling2d_layer_call_fn_9297144?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84????????????????????????????????????