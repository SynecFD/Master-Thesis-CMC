??
??
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
 ?"serve*2.4.12unknown8??
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
x
training/Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *#
shared_nametraining/Adam/iter
q
&training/Adam/iter/Read/ReadVariableOpReadVariableOptraining/Adam/iter*
_output_shapes
: *
dtype0	
|
training/Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nametraining/Adam/beta_1
u
(training/Adam/beta_1/Read/ReadVariableOpReadVariableOptraining/Adam/beta_1*
_output_shapes
: *
dtype0
|
training/Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nametraining/Adam/beta_2
u
(training/Adam/beta_2/Read/ReadVariableOpReadVariableOptraining/Adam/beta_2*
_output_shapes
: *
dtype0
z
training/Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nametraining/Adam/decay
s
'training/Adam/decay/Read/ReadVariableOpReadVariableOptraining/Adam/decay*
_output_shapes
: *
dtype0
?
training/Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nametraining/Adam/learning_rate
?
/training/Adam/learning_rate/Read/ReadVariableOpReadVariableOptraining/Adam/learning_rate*
_output_shapes
: *
dtype0
?
training/Adam/conv1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nametraining/Adam/conv1/kernel/m
?
0training/Adam/conv1/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv1/kernel/m*&
_output_shapes
:*
dtype0
?
training/Adam/conv1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nametraining/Adam/conv1/bias/m
?
.training/Adam/conv1/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv1/bias/m*
_output_shapes
:*
dtype0
?
training/Adam/conv2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nametraining/Adam/conv2/kernel/m
?
0training/Adam/conv2/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv2/kernel/m*&
_output_shapes
:*
dtype0
?
training/Adam/conv2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nametraining/Adam/conv2/bias/m
?
.training/Adam/conv2/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/conv2/bias/m*
_output_shapes
:*
dtype0
?
training/Adam/dense1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *.
shared_nametraining/Adam/dense1/kernel/m
?
1training/Adam/dense1/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense1/kernel/m*
_output_shapes
:	? *
dtype0
?
training/Adam/dense1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nametraining/Adam/dense1/bias/m
?
/training/Adam/dense1/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense1/bias/m*
_output_shapes
: *
dtype0
?
training/Adam/dense2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*.
shared_nametraining/Adam/dense2/kernel/m
?
1training/Adam/dense2/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense2/kernel/m*
_output_shapes
:	 ?*
dtype0
?
training/Adam/dense2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_nametraining/Adam/dense2/bias/m
?
/training/Adam/dense2/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense2/bias/m*
_output_shapes	
:?*
dtype0
?
 training/Adam/dec_conv1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" training/Adam/dec_conv1/kernel/m
?
4training/Adam/dec_conv1/kernel/m/Read/ReadVariableOpReadVariableOp training/Adam/dec_conv1/kernel/m*&
_output_shapes
:*
dtype0
?
training/Adam/dec_conv1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name training/Adam/dec_conv1/bias/m
?
2training/Adam/dec_conv1/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/dec_conv1/bias/m*
_output_shapes
:*
dtype0
?
 training/Adam/dec_conv2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*1
shared_name" training/Adam/dec_conv2/kernel/m
?
4training/Adam/dec_conv2/kernel/m/Read/ReadVariableOpReadVariableOp training/Adam/dec_conv2/kernel/m*&
_output_shapes
:
*
dtype0
?
training/Adam/dec_conv2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name training/Adam/dec_conv2/bias/m
?
2training/Adam/dec_conv2/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/dec_conv2/bias/m*
_output_shapes
:*
dtype0
?
training/Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:#*/
shared_name training/Adam/dense_2/kernel/m
?
2training/Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_2/kernel/m*
_output_shapes

:#*
dtype0
?
training/Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nametraining/Adam/dense_2/bias/m
?
0training/Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_2/bias/m*
_output_shapes
:*
dtype0
?
 training/Adam/dec_conv3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" training/Adam/dec_conv3/kernel/m
?
4training/Adam/dec_conv3/kernel/m/Read/ReadVariableOpReadVariableOp training/Adam/dec_conv3/kernel/m*&
_output_shapes
:*
dtype0
?
training/Adam/dec_conv3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name training/Adam/dec_conv3/bias/m
?
2training/Adam/dec_conv3/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/dec_conv3/bias/m*
_output_shapes
:*
dtype0
?
training/Adam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*/
shared_name training/Adam/dense_3/kernel/m
?
2training/Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_3/kernel/m*
_output_shapes

:*
dtype0
?
training/Adam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nametraining/Adam/dense_3/bias/m
?
0training/Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOptraining/Adam/dense_3/bias/m*
_output_shapes
:*
dtype0
?
training/Adam/conv1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nametraining/Adam/conv1/kernel/v
?
0training/Adam/conv1/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv1/kernel/v*&
_output_shapes
:*
dtype0
?
training/Adam/conv1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nametraining/Adam/conv1/bias/v
?
.training/Adam/conv1/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv1/bias/v*
_output_shapes
:*
dtype0
?
training/Adam/conv2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nametraining/Adam/conv2/kernel/v
?
0training/Adam/conv2/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv2/kernel/v*&
_output_shapes
:*
dtype0
?
training/Adam/conv2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nametraining/Adam/conv2/bias/v
?
.training/Adam/conv2/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/conv2/bias/v*
_output_shapes
:*
dtype0
?
training/Adam/dense1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? *.
shared_nametraining/Adam/dense1/kernel/v
?
1training/Adam/dense1/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense1/kernel/v*
_output_shapes
:	? *
dtype0
?
training/Adam/dense1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nametraining/Adam/dense1/bias/v
?
/training/Adam/dense1/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense1/bias/v*
_output_shapes
: *
dtype0
?
training/Adam/dense2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	 ?*.
shared_nametraining/Adam/dense2/kernel/v
?
1training/Adam/dense2/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense2/kernel/v*
_output_shapes
:	 ?*
dtype0
?
training/Adam/dense2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_nametraining/Adam/dense2/bias/v
?
/training/Adam/dense2/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense2/bias/v*
_output_shapes	
:?*
dtype0
?
 training/Adam/dec_conv1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" training/Adam/dec_conv1/kernel/v
?
4training/Adam/dec_conv1/kernel/v/Read/ReadVariableOpReadVariableOp training/Adam/dec_conv1/kernel/v*&
_output_shapes
:*
dtype0
?
training/Adam/dec_conv1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name training/Adam/dec_conv1/bias/v
?
2training/Adam/dec_conv1/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/dec_conv1/bias/v*
_output_shapes
:*
dtype0
?
 training/Adam/dec_conv2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*1
shared_name" training/Adam/dec_conv2/kernel/v
?
4training/Adam/dec_conv2/kernel/v/Read/ReadVariableOpReadVariableOp training/Adam/dec_conv2/kernel/v*&
_output_shapes
:
*
dtype0
?
training/Adam/dec_conv2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name training/Adam/dec_conv2/bias/v
?
2training/Adam/dec_conv2/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/dec_conv2/bias/v*
_output_shapes
:*
dtype0
?
training/Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:#*/
shared_name training/Adam/dense_2/kernel/v
?
2training/Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_2/kernel/v*
_output_shapes

:#*
dtype0
?
training/Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nametraining/Adam/dense_2/bias/v
?
0training/Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_2/bias/v*
_output_shapes
:*
dtype0
?
 training/Adam/dec_conv3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" training/Adam/dec_conv3/kernel/v
?
4training/Adam/dec_conv3/kernel/v/Read/ReadVariableOpReadVariableOp training/Adam/dec_conv3/kernel/v*&
_output_shapes
:*
dtype0
?
training/Adam/dec_conv3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name training/Adam/dec_conv3/bias/v
?
2training/Adam/dec_conv3/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/dec_conv3/bias/v*
_output_shapes
:*
dtype0
?
training/Adam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*/
shared_name training/Adam/dense_3/kernel/v
?
2training/Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_3/kernel/v*
_output_shapes

:*
dtype0
?
training/Adam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nametraining/Adam/dense_3/bias/v
?
0training/Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOptraining/Adam/dense_3/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?j
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?j
value?jB?j B?j
?
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

layer-9
layer_with_weights-5
layer-10
layer-11
layer-12
layer_with_weights-6
layer-13
layer_with_weights-7
layer-14
layer_with_weights-8
layer-15
	optimizer

signatures
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
%
#_self_saveable_object_factories
?

kernel
bias
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
?

 kernel
!bias
#"_self_saveable_object_factories
#regularization_losses
$	variables
%trainable_variables
&	keras_api
w
#'_self_saveable_object_factories
(regularization_losses
)	variables
*trainable_variables
+	keras_api
?

,kernel
-bias
#._self_saveable_object_factories
/regularization_losses
0	variables
1trainable_variables
2	keras_api
?

3kernel
4bias
#5_self_saveable_object_factories
6regularization_losses
7	variables
8trainable_variables
9	keras_api
w
#:_self_saveable_object_factories
;regularization_losses
<	variables
=trainable_variables
>	keras_api
?

?kernel
@bias
#A_self_saveable_object_factories
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
w
#F_self_saveable_object_factories
Gregularization_losses
H	variables
Itrainable_variables
J	keras_api
%
#K_self_saveable_object_factories
?

Lkernel
Mbias
#N_self_saveable_object_factories
Oregularization_losses
P	variables
Qtrainable_variables
R	keras_api
w
#S_self_saveable_object_factories
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
w
#X_self_saveable_object_factories
Yregularization_losses
Z	variables
[trainable_variables
\	keras_api
?

]kernel
^bias
#__self_saveable_object_factories
`regularization_losses
a	variables
btrainable_variables
c	keras_api
?

dkernel
ebias
#f_self_saveable_object_factories
gregularization_losses
h	variables
itrainable_variables
j	keras_api
?

kkernel
lbias
#m_self_saveable_object_factories
nregularization_losses
o	variables
ptrainable_variables
q	keras_api
?
riter

sbeta_1

tbeta_2
	udecay
vlearning_ratem?m? m?!m?,m?-m?3m?4m??m?@m?Lm?Mm?]m?^m?dm?em?km?lm?v?v? v?!v?,v?-v?3v?4v??v?@v?Lv?Mv?]v?^v?dv?ev?kv?lv?
 
 
 
?
0
1
 2
!3
,4
-5
36
47
?8
@9
L10
M11
]12
^13
d14
e15
k16
l17
?
0
1
 2
!3
,4
-5
36
47
?8
@9
L10
M11
]12
^13
d14
e15
k16
l17
?
wlayer_metrics
regularization_losses
xlayer_regularization_losses
ynon_trainable_variables

zlayers
	variables
{metrics
trainable_variables
 
XV
VARIABLE_VALUEconv1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
?
|layer_metrics
}layer_regularization_losses
regularization_losses
~non_trainable_variables

layers
	variables
?metrics
trainable_variables
XV
VARIABLE_VALUEconv2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
conv2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

 0
!1

 0
!1
?
?layer_metrics
 ?layer_regularization_losses
#regularization_losses
?non_trainable_variables
?layers
$	variables
?metrics
%trainable_variables
 
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
(regularization_losses
?non_trainable_variables
?layers
)	variables
?metrics
*trainable_variables
YW
VARIABLE_VALUEdense1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

,0
-1

,0
-1
?
?layer_metrics
 ?layer_regularization_losses
/regularization_losses
?non_trainable_variables
?layers
0	variables
?metrics
1trainable_variables
YW
VARIABLE_VALUEdense2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

30
41

30
41
?
?layer_metrics
 ?layer_regularization_losses
6regularization_losses
?non_trainable_variables
?layers
7	variables
?metrics
8trainable_variables
 
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
;regularization_losses
?non_trainable_variables
?layers
<	variables
?metrics
=trainable_variables
\Z
VARIABLE_VALUEdec_conv1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdec_conv1/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

?0
@1

?0
@1
?
?layer_metrics
 ?layer_regularization_losses
Bregularization_losses
?non_trainable_variables
?layers
C	variables
?metrics
Dtrainable_variables
 
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
Gregularization_losses
?non_trainable_variables
?layers
H	variables
?metrics
Itrainable_variables
 
\Z
VARIABLE_VALUEdec_conv2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdec_conv2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

L0
M1

L0
M1
?
?layer_metrics
 ?layer_regularization_losses
Oregularization_losses
?non_trainable_variables
?layers
P	variables
?metrics
Qtrainable_variables
 
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
Tregularization_losses
?non_trainable_variables
?layers
U	variables
?metrics
Vtrainable_variables
 
 
 
 
?
?layer_metrics
 ?layer_regularization_losses
Yregularization_losses
?non_trainable_variables
?layers
Z	variables
?metrics
[trainable_variables
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

]0
^1

]0
^1
?
?layer_metrics
 ?layer_regularization_losses
`regularization_losses
?non_trainable_variables
?layers
a	variables
?metrics
btrainable_variables
\Z
VARIABLE_VALUEdec_conv3/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEdec_conv3/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

d0
e1

d0
e1
?
?layer_metrics
 ?layer_regularization_losses
gregularization_losses
?non_trainable_variables
?layers
h	variables
?metrics
itrainable_variables
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

k0
l1

k0
l1
?
?layer_metrics
 ?layer_regularization_losses
nregularization_losses
?non_trainable_variables
?layers
o	variables
?metrics
ptrainable_variables
QO
VARIABLE_VALUEtraining/Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEtraining/Adam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEtraining/Adam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEtraining/Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEtraining/Adam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
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
11
12
13
14
15
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
VARIABLE_VALUEtraining/Adam/conv1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEtraining/Adam/conv1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/conv2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEtraining/Adam/conv2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/dense1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEtraining/Adam/dense1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/dense2/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEtraining/Adam/dense2/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE training/Adam/dec_conv1/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/dec_conv1/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE training/Adam/dec_conv2/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/dec_conv2/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/dense_2/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/dense_2/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE training/Adam/dec_conv3/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/dec_conv3/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/dense_3/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/dense_3/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/conv1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEtraining/Adam/conv1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/conv2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUEtraining/Adam/conv2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/dense1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEtraining/Adam/dense1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/dense2/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEtraining/Adam/dense2/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE training/Adam/dec_conv1/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/dec_conv1/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE training/Adam/dec_conv2/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/dec_conv2/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/dense_2/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/dense_2/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE training/Adam/dec_conv3/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/dec_conv3/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/dense_3/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEtraining/Adam/dense_3/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
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
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_actserving_default_input_imgconv1/kernel
conv1/biasconv2/kernel
conv2/biasdense1/kerneldense1/biasdense2/kerneldense2/biasdec_conv1/kerneldec_conv1/biasdec_conv2/kerneldec_conv2/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdec_conv3/kerneldec_conv3/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:?????????@@:?????????*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8? *.
f)R'
%__inference_signature_wrapper_9295524
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename conv1/kernel/Read/ReadVariableOpconv1/bias/Read/ReadVariableOp conv2/kernel/Read/ReadVariableOpconv2/bias/Read/ReadVariableOp!dense1/kernel/Read/ReadVariableOpdense1/bias/Read/ReadVariableOp!dense2/kernel/Read/ReadVariableOpdense2/bias/Read/ReadVariableOp$dec_conv1/kernel/Read/ReadVariableOp"dec_conv1/bias/Read/ReadVariableOp$dec_conv2/kernel/Read/ReadVariableOp"dec_conv2/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp$dec_conv3/kernel/Read/ReadVariableOp"dec_conv3/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp&training/Adam/iter/Read/ReadVariableOp(training/Adam/beta_1/Read/ReadVariableOp(training/Adam/beta_2/Read/ReadVariableOp'training/Adam/decay/Read/ReadVariableOp/training/Adam/learning_rate/Read/ReadVariableOp0training/Adam/conv1/kernel/m/Read/ReadVariableOp.training/Adam/conv1/bias/m/Read/ReadVariableOp0training/Adam/conv2/kernel/m/Read/ReadVariableOp.training/Adam/conv2/bias/m/Read/ReadVariableOp1training/Adam/dense1/kernel/m/Read/ReadVariableOp/training/Adam/dense1/bias/m/Read/ReadVariableOp1training/Adam/dense2/kernel/m/Read/ReadVariableOp/training/Adam/dense2/bias/m/Read/ReadVariableOp4training/Adam/dec_conv1/kernel/m/Read/ReadVariableOp2training/Adam/dec_conv1/bias/m/Read/ReadVariableOp4training/Adam/dec_conv2/kernel/m/Read/ReadVariableOp2training/Adam/dec_conv2/bias/m/Read/ReadVariableOp2training/Adam/dense_2/kernel/m/Read/ReadVariableOp0training/Adam/dense_2/bias/m/Read/ReadVariableOp4training/Adam/dec_conv3/kernel/m/Read/ReadVariableOp2training/Adam/dec_conv3/bias/m/Read/ReadVariableOp2training/Adam/dense_3/kernel/m/Read/ReadVariableOp0training/Adam/dense_3/bias/m/Read/ReadVariableOp0training/Adam/conv1/kernel/v/Read/ReadVariableOp.training/Adam/conv1/bias/v/Read/ReadVariableOp0training/Adam/conv2/kernel/v/Read/ReadVariableOp.training/Adam/conv2/bias/v/Read/ReadVariableOp1training/Adam/dense1/kernel/v/Read/ReadVariableOp/training/Adam/dense1/bias/v/Read/ReadVariableOp1training/Adam/dense2/kernel/v/Read/ReadVariableOp/training/Adam/dense2/bias/v/Read/ReadVariableOp4training/Adam/dec_conv1/kernel/v/Read/ReadVariableOp2training/Adam/dec_conv1/bias/v/Read/ReadVariableOp4training/Adam/dec_conv2/kernel/v/Read/ReadVariableOp2training/Adam/dec_conv2/bias/v/Read/ReadVariableOp2training/Adam/dense_2/kernel/v/Read/ReadVariableOp0training/Adam/dense_2/bias/v/Read/ReadVariableOp4training/Adam/dec_conv3/kernel/v/Read/ReadVariableOp2training/Adam/dec_conv3/bias/v/Read/ReadVariableOp2training/Adam/dense_3/kernel/v/Read/ReadVariableOp0training/Adam/dense_3/bias/v/Read/ReadVariableOpConst*H
TinA
?2=	*
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
 __inference__traced_save_9296178
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1/kernel
conv1/biasconv2/kernel
conv2/biasdense1/kerneldense1/biasdense2/kerneldense2/biasdec_conv1/kerneldec_conv1/biasdec_conv2/kerneldec_conv2/biasdense_2/kerneldense_2/biasdec_conv3/kerneldec_conv3/biasdense_3/kerneldense_3/biastraining/Adam/itertraining/Adam/beta_1training/Adam/beta_2training/Adam/decaytraining/Adam/learning_ratetraining/Adam/conv1/kernel/mtraining/Adam/conv1/bias/mtraining/Adam/conv2/kernel/mtraining/Adam/conv2/bias/mtraining/Adam/dense1/kernel/mtraining/Adam/dense1/bias/mtraining/Adam/dense2/kernel/mtraining/Adam/dense2/bias/m training/Adam/dec_conv1/kernel/mtraining/Adam/dec_conv1/bias/m training/Adam/dec_conv2/kernel/mtraining/Adam/dec_conv2/bias/mtraining/Adam/dense_2/kernel/mtraining/Adam/dense_2/bias/m training/Adam/dec_conv3/kernel/mtraining/Adam/dec_conv3/bias/mtraining/Adam/dense_3/kernel/mtraining/Adam/dense_3/bias/mtraining/Adam/conv1/kernel/vtraining/Adam/conv1/bias/vtraining/Adam/conv2/kernel/vtraining/Adam/conv2/bias/vtraining/Adam/dense1/kernel/vtraining/Adam/dense1/bias/vtraining/Adam/dense2/kernel/vtraining/Adam/dense2/bias/v training/Adam/dec_conv1/kernel/vtraining/Adam/dec_conv1/bias/v training/Adam/dec_conv2/kernel/vtraining/Adam/dec_conv2/bias/vtraining/Adam/dense_2/kernel/vtraining/Adam/dense_2/bias/v training/Adam/dec_conv3/kernel/vtraining/Adam/dec_conv3/bias/vtraining/Adam/dense_3/kernel/vtraining/Adam/dense_3/bias/v*G
Tin@
>2<*
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
#__inference__traced_restore_9296365??
?

?
B__inference_conv2_layer_call_and_return_conditional_losses_9295801

inputs"
conv2d_readvariableop_kernel_1!
biasadd_readvariableop_bias_1
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_kernel_1*&
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
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_1*
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
C__inference_dense1_layer_call_and_return_conditional_losses_9295128

inputs"
matmul_readvariableop_kernel_2!
biasadd_readvariableop_bias_2
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_kernel_2*
_output_shapes
:	? *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_2*
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
??
?
"__inference__wrapped_model_9294990
	input_img
	input_act.
*model_5_conv1_conv2d_readvariableop_kernel-
)model_5_conv1_biasadd_readvariableop_bias0
,model_5_conv2_conv2d_readvariableop_kernel_1/
+model_5_conv2_biasadd_readvariableop_bias_11
-model_5_dense1_matmul_readvariableop_kernel_20
,model_5_dense1_biasadd_readvariableop_bias_21
-model_5_dense2_matmul_readvariableop_kernel_30
,model_5_dense2_biasadd_readvariableop_bias_34
0model_5_dec_conv1_conv2d_readvariableop_kernel_43
/model_5_dec_conv1_biasadd_readvariableop_bias_44
0model_5_dec_conv2_conv2d_readvariableop_kernel_53
/model_5_dec_conv2_biasadd_readvariableop_bias_52
.model_5_dense_2_matmul_readvariableop_kernel_61
-model_5_dense_2_biasadd_readvariableop_bias_62
.model_5_dense_3_matmul_readvariableop_kernel_81
-model_5_dense_3_biasadd_readvariableop_bias_84
0model_5_dec_conv3_conv2d_readvariableop_kernel_73
/model_5_dec_conv3_biasadd_readvariableop_bias_7
identity

identity_1??$model_5/conv1/BiasAdd/ReadVariableOp?#model_5/conv1/Conv2D/ReadVariableOp?$model_5/conv2/BiasAdd/ReadVariableOp?#model_5/conv2/Conv2D/ReadVariableOp?(model_5/dec_conv1/BiasAdd/ReadVariableOp?'model_5/dec_conv1/Conv2D/ReadVariableOp?(model_5/dec_conv2/BiasAdd/ReadVariableOp?'model_5/dec_conv2/Conv2D/ReadVariableOp?(model_5/dec_conv3/BiasAdd/ReadVariableOp?'model_5/dec_conv3/Conv2D/ReadVariableOp?%model_5/dense1/BiasAdd/ReadVariableOp?$model_5/dense1/MatMul/ReadVariableOp?%model_5/dense2/BiasAdd/ReadVariableOp?$model_5/dense2/MatMul/ReadVariableOp?&model_5/dense_2/BiasAdd/ReadVariableOp?%model_5/dense_2/MatMul/ReadVariableOp?&model_5/dense_3/BiasAdd/ReadVariableOp?%model_5/dense_3/MatMul/ReadVariableOp?
#model_5/conv1/Conv2D/ReadVariableOpReadVariableOp*model_5_conv1_conv2d_readvariableop_kernel*&
_output_shapes
:*
dtype02%
#model_5/conv1/Conv2D/ReadVariableOp?
model_5/conv1/Conv2DConv2D	input_img+model_5/conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
model_5/conv1/Conv2D?
$model_5/conv1/BiasAdd/ReadVariableOpReadVariableOp)model_5_conv1_biasadd_readvariableop_bias*
_output_shapes
:*
dtype02&
$model_5/conv1/BiasAdd/ReadVariableOp?
model_5/conv1/BiasAddBiasAddmodel_5/conv1/Conv2D:output:0,model_5/conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
model_5/conv1/BiasAdd?
model_5/conv1/ReluRelumodel_5/conv1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
model_5/conv1/Relu?
#model_5/conv2/Conv2D/ReadVariableOpReadVariableOp,model_5_conv2_conv2d_readvariableop_kernel_1*&
_output_shapes
:*
dtype02%
#model_5/conv2/Conv2D/ReadVariableOp?
model_5/conv2/Conv2DConv2D model_5/conv1/Relu:activations:0+model_5/conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
model_5/conv2/Conv2D?
$model_5/conv2/BiasAdd/ReadVariableOpReadVariableOp+model_5_conv2_biasadd_readvariableop_bias_1*
_output_shapes
:*
dtype02&
$model_5/conv2/BiasAdd/ReadVariableOp?
model_5/conv2/BiasAddBiasAddmodel_5/conv2/Conv2D:output:0,model_5/conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
model_5/conv2/BiasAdd?
model_5/conv2/ReluRelumodel_5/conv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
model_5/conv2/Relu
model_5/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  2
model_5/flatten/Const?
model_5/flatten/ReshapeReshape model_5/conv2/Relu:activations:0model_5/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
model_5/flatten/Reshape?
$model_5/dense1/MatMul/ReadVariableOpReadVariableOp-model_5_dense1_matmul_readvariableop_kernel_2*
_output_shapes
:	? *
dtype02&
$model_5/dense1/MatMul/ReadVariableOp?
model_5/dense1/MatMulMatMul model_5/flatten/Reshape:output:0,model_5/dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
model_5/dense1/MatMul?
%model_5/dense1/BiasAdd/ReadVariableOpReadVariableOp,model_5_dense1_biasadd_readvariableop_bias_2*
_output_shapes
: *
dtype02'
%model_5/dense1/BiasAdd/ReadVariableOp?
model_5/dense1/BiasAddBiasAddmodel_5/dense1/MatMul:product:0-model_5/dense1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
model_5/dense1/BiasAdd?
model_5/dense1/ReluRelumodel_5/dense1/BiasAdd:output:0*
T0*'
_output_shapes
:????????? 2
model_5/dense1/Relu?
$model_5/dense2/MatMul/ReadVariableOpReadVariableOp-model_5_dense2_matmul_readvariableop_kernel_3*
_output_shapes
:	 ?*
dtype02&
$model_5/dense2/MatMul/ReadVariableOp?
model_5/dense2/MatMulMatMul!model_5/dense1/Relu:activations:0,model_5/dense2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_5/dense2/MatMul?
%model_5/dense2/BiasAdd/ReadVariableOpReadVariableOp,model_5_dense2_biasadd_readvariableop_bias_3*
_output_shapes	
:?*
dtype02'
%model_5/dense2/BiasAdd/ReadVariableOp?
model_5/dense2/BiasAddBiasAddmodel_5/dense2/MatMul:product:0-model_5/dense2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
model_5/dense2/BiasAdd?
model_5/dense2/ReluRelumodel_5/dense2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
model_5/dense2/Relu
model_5/reshape/ShapeShape!model_5/dense2/Relu:activations:0*
T0*
_output_shapes
:2
model_5/reshape/Shape?
#model_5/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model_5/reshape/strided_slice/stack?
%model_5/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%model_5/reshape/strided_slice/stack_1?
%model_5/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%model_5/reshape/strided_slice/stack_2?
model_5/reshape/strided_sliceStridedSlicemodel_5/reshape/Shape:output:0,model_5/reshape/strided_slice/stack:output:0.model_5/reshape/strided_slice/stack_1:output:0.model_5/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model_5/reshape/strided_slice?
model_5/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2!
model_5/reshape/Reshape/shape/1?
model_5/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2!
model_5/reshape/Reshape/shape/2?
model_5/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2!
model_5/reshape/Reshape/shape/3?
model_5/reshape/Reshape/shapePack&model_5/reshape/strided_slice:output:0(model_5/reshape/Reshape/shape/1:output:0(model_5/reshape/Reshape/shape/2:output:0(model_5/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
model_5/reshape/Reshape/shape?
model_5/reshape/ReshapeReshape!model_5/dense2/Relu:activations:0&model_5/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
model_5/reshape/Reshape?
'model_5/dec_conv1/Conv2D/ReadVariableOpReadVariableOp0model_5_dec_conv1_conv2d_readvariableop_kernel_4*&
_output_shapes
:*
dtype02)
'model_5/dec_conv1/Conv2D/ReadVariableOp?
model_5/dec_conv1/Conv2DConv2D model_5/reshape/Reshape:output:0/model_5/dec_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
model_5/dec_conv1/Conv2D?
(model_5/dec_conv1/BiasAdd/ReadVariableOpReadVariableOp/model_5_dec_conv1_biasadd_readvariableop_bias_4*
_output_shapes
:*
dtype02*
(model_5/dec_conv1/BiasAdd/ReadVariableOp?
model_5/dec_conv1/BiasAddBiasAdd!model_5/dec_conv1/Conv2D:output:00model_5/dec_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
model_5/dec_conv1/BiasAdd?
model_5/dec_conv1/ReluRelu"model_5/dec_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
model_5/dec_conv1/Relu?
model_5/up_sampling2d/ShapeShape$model_5/dec_conv1/Relu:activations:0*
T0*
_output_shapes
:2
model_5/up_sampling2d/Shape?
)model_5/up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)model_5/up_sampling2d/strided_slice/stack?
+model_5/up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+model_5/up_sampling2d/strided_slice/stack_1?
+model_5/up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+model_5/up_sampling2d/strided_slice/stack_2?
#model_5/up_sampling2d/strided_sliceStridedSlice$model_5/up_sampling2d/Shape:output:02model_5/up_sampling2d/strided_slice/stack:output:04model_5/up_sampling2d/strided_slice/stack_1:output:04model_5/up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2%
#model_5/up_sampling2d/strided_slice?
model_5/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
model_5/up_sampling2d/Const?
model_5/up_sampling2d/mulMul,model_5/up_sampling2d/strided_slice:output:0$model_5/up_sampling2d/Const:output:0*
T0*
_output_shapes
:2
model_5/up_sampling2d/mul?
2model_5/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor$model_5/dec_conv1/Relu:activations:0model_5/up_sampling2d/mul:z:0*
T0*/
_output_shapes
:?????????*
half_pixel_centers(24
2model_5/up_sampling2d/resize/ResizeNearestNeighbor?
!model_5/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_5/concatenate_2/concat/axis?
model_5/concatenate_2/concatConcatV2!model_5/dense1/Relu:activations:0	input_act*model_5/concatenate_2/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????#2
model_5/concatenate_2/concat?
'model_5/dec_conv2/Conv2D/ReadVariableOpReadVariableOp0model_5_dec_conv2_conv2d_readvariableop_kernel_5*&
_output_shapes
:
*
dtype02)
'model_5/dec_conv2/Conv2D/ReadVariableOp?
model_5/dec_conv2/Conv2DConv2DCmodel_5/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0/model_5/dec_conv2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
model_5/dec_conv2/Conv2D?
(model_5/dec_conv2/BiasAdd/ReadVariableOpReadVariableOp/model_5_dec_conv2_biasadd_readvariableop_bias_5*
_output_shapes
:*
dtype02*
(model_5/dec_conv2/BiasAdd/ReadVariableOp?
model_5/dec_conv2/BiasAddBiasAdd!model_5/dec_conv2/Conv2D:output:00model_5/dec_conv2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
model_5/dec_conv2/BiasAdd?
model_5/dec_conv2/ReluRelu"model_5/dec_conv2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
model_5/dec_conv2/Relu?
%model_5/dense_2/MatMul/ReadVariableOpReadVariableOp.model_5_dense_2_matmul_readvariableop_kernel_6*
_output_shapes

:#*
dtype02'
%model_5/dense_2/MatMul/ReadVariableOp?
model_5/dense_2/MatMulMatMul%model_5/concatenate_2/concat:output:0-model_5/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_5/dense_2/MatMul?
&model_5/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_5_dense_2_biasadd_readvariableop_bias_6*
_output_shapes
:*
dtype02(
&model_5/dense_2/BiasAdd/ReadVariableOp?
model_5/dense_2/BiasAddBiasAdd model_5/dense_2/MatMul:product:0.model_5/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_5/dense_2/BiasAdd?
model_5/dense_2/ReluRelu model_5/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_5/dense_2/Relu?
model_5/up_sampling2d_1/ShapeShape$model_5/dec_conv2/Relu:activations:0*
T0*
_output_shapes
:2
model_5/up_sampling2d_1/Shape?
+model_5/up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+model_5/up_sampling2d_1/strided_slice/stack?
-model_5/up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-model_5/up_sampling2d_1/strided_slice/stack_1?
-model_5/up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-model_5/up_sampling2d_1/strided_slice/stack_2?
%model_5/up_sampling2d_1/strided_sliceStridedSlice&model_5/up_sampling2d_1/Shape:output:04model_5/up_sampling2d_1/strided_slice/stack:output:06model_5/up_sampling2d_1/strided_slice/stack_1:output:06model_5/up_sampling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2'
%model_5/up_sampling2d_1/strided_slice?
model_5/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
model_5/up_sampling2d_1/Const?
model_5/up_sampling2d_1/mulMul.model_5/up_sampling2d_1/strided_slice:output:0&model_5/up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2
model_5/up_sampling2d_1/mul?
4model_5/up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor$model_5/dec_conv2/Relu:activations:0model_5/up_sampling2d_1/mul:z:0*
T0*/
_output_shapes
:?????????KK*
half_pixel_centers(26
4model_5/up_sampling2d_1/resize/ResizeNearestNeighbor?
%model_5/dense_3/MatMul/ReadVariableOpReadVariableOp.model_5_dense_3_matmul_readvariableop_kernel_8*
_output_shapes

:*
dtype02'
%model_5/dense_3/MatMul/ReadVariableOp?
model_5/dense_3/MatMulMatMul"model_5/dense_2/Relu:activations:0-model_5/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_5/dense_3/MatMul?
&model_5/dense_3/BiasAdd/ReadVariableOpReadVariableOp-model_5_dense_3_biasadd_readvariableop_bias_8*
_output_shapes
:*
dtype02(
&model_5/dense_3/BiasAdd/ReadVariableOp?
model_5/dense_3/BiasAddBiasAdd model_5/dense_3/MatMul:product:0.model_5/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_5/dense_3/BiasAdd?
'model_5/dec_conv3/Conv2D/ReadVariableOpReadVariableOp0model_5_dec_conv3_conv2d_readvariableop_kernel_7*&
_output_shapes
:*
dtype02)
'model_5/dec_conv3/Conv2D/ReadVariableOp?
model_5/dec_conv3/Conv2DConv2DEmodel_5/up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0/model_5/dec_conv3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@*
paddingVALID*
strides
2
model_5/dec_conv3/Conv2D?
(model_5/dec_conv3/BiasAdd/ReadVariableOpReadVariableOp/model_5_dec_conv3_biasadd_readvariableop_bias_7*
_output_shapes
:*
dtype02*
(model_5/dec_conv3/BiasAdd/ReadVariableOp?
model_5/dec_conv3/BiasAddBiasAdd!model_5/dec_conv3/Conv2D:output:00model_5/dec_conv3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@2
model_5/dec_conv3/BiasAdd?
model_5/dec_conv3/ReluRelu"model_5/dec_conv3/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@2
model_5/dec_conv3/Relu?
IdentityIdentity$model_5/dec_conv3/Relu:activations:0%^model_5/conv1/BiasAdd/ReadVariableOp$^model_5/conv1/Conv2D/ReadVariableOp%^model_5/conv2/BiasAdd/ReadVariableOp$^model_5/conv2/Conv2D/ReadVariableOp)^model_5/dec_conv1/BiasAdd/ReadVariableOp(^model_5/dec_conv1/Conv2D/ReadVariableOp)^model_5/dec_conv2/BiasAdd/ReadVariableOp(^model_5/dec_conv2/Conv2D/ReadVariableOp)^model_5/dec_conv3/BiasAdd/ReadVariableOp(^model_5/dec_conv3/Conv2D/ReadVariableOp&^model_5/dense1/BiasAdd/ReadVariableOp%^model_5/dense1/MatMul/ReadVariableOp&^model_5/dense2/BiasAdd/ReadVariableOp%^model_5/dense2/MatMul/ReadVariableOp'^model_5/dense_2/BiasAdd/ReadVariableOp&^model_5/dense_2/MatMul/ReadVariableOp'^model_5/dense_3/BiasAdd/ReadVariableOp&^model_5/dense_3/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????@@2

Identity?

Identity_1Identity model_5/dense_3/BiasAdd:output:0%^model_5/conv1/BiasAdd/ReadVariableOp$^model_5/conv1/Conv2D/ReadVariableOp%^model_5/conv2/BiasAdd/ReadVariableOp$^model_5/conv2/Conv2D/ReadVariableOp)^model_5/dec_conv1/BiasAdd/ReadVariableOp(^model_5/dec_conv1/Conv2D/ReadVariableOp)^model_5/dec_conv2/BiasAdd/ReadVariableOp(^model_5/dec_conv2/Conv2D/ReadVariableOp)^model_5/dec_conv3/BiasAdd/ReadVariableOp(^model_5/dec_conv3/Conv2D/ReadVariableOp&^model_5/dense1/BiasAdd/ReadVariableOp%^model_5/dense1/MatMul/ReadVariableOp&^model_5/dense2/BiasAdd/ReadVariableOp%^model_5/dense2/MatMul/ReadVariableOp'^model_5/dense_2/BiasAdd/ReadVariableOp&^model_5/dense_2/MatMul/ReadVariableOp'^model_5/dense_3/BiasAdd/ReadVariableOp&^model_5/dense_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapesx
v:?????????@@:?????????::::::::::::::::::2L
$model_5/conv1/BiasAdd/ReadVariableOp$model_5/conv1/BiasAdd/ReadVariableOp2J
#model_5/conv1/Conv2D/ReadVariableOp#model_5/conv1/Conv2D/ReadVariableOp2L
$model_5/conv2/BiasAdd/ReadVariableOp$model_5/conv2/BiasAdd/ReadVariableOp2J
#model_5/conv2/Conv2D/ReadVariableOp#model_5/conv2/Conv2D/ReadVariableOp2T
(model_5/dec_conv1/BiasAdd/ReadVariableOp(model_5/dec_conv1/BiasAdd/ReadVariableOp2R
'model_5/dec_conv1/Conv2D/ReadVariableOp'model_5/dec_conv1/Conv2D/ReadVariableOp2T
(model_5/dec_conv2/BiasAdd/ReadVariableOp(model_5/dec_conv2/BiasAdd/ReadVariableOp2R
'model_5/dec_conv2/Conv2D/ReadVariableOp'model_5/dec_conv2/Conv2D/ReadVariableOp2T
(model_5/dec_conv3/BiasAdd/ReadVariableOp(model_5/dec_conv3/BiasAdd/ReadVariableOp2R
'model_5/dec_conv3/Conv2D/ReadVariableOp'model_5/dec_conv3/Conv2D/ReadVariableOp2N
%model_5/dense1/BiasAdd/ReadVariableOp%model_5/dense1/BiasAdd/ReadVariableOp2L
$model_5/dense1/MatMul/ReadVariableOp$model_5/dense1/MatMul/ReadVariableOp2N
%model_5/dense2/BiasAdd/ReadVariableOp%model_5/dense2/BiasAdd/ReadVariableOp2L
$model_5/dense2/MatMul/ReadVariableOp$model_5/dense2/MatMul/ReadVariableOp2P
&model_5/dense_2/BiasAdd/ReadVariableOp&model_5/dense_2/BiasAdd/ReadVariableOp2N
%model_5/dense_2/MatMul/ReadVariableOp%model_5/dense_2/MatMul/ReadVariableOp2P
&model_5/dense_3/BiasAdd/ReadVariableOp&model_5/dense_3/BiasAdd/ReadVariableOp2N
%model_5/dense_3/MatMul/ReadVariableOp%model_5/dense_3/MatMul/ReadVariableOp:Z V
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
`
D__inference_reshape_layer_call_and_return_conditional_losses_9295177

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
?
E
)__inference_reshape_layer_call_fn_9295874

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
D__inference_reshape_layer_call_and_return_conditional_losses_92951772
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
?D
?
D__inference_model_5_layer_call_and_return_conditional_losses_9295367
	input_img
	input_act
conv1_kernel

conv1_bias
conv2_kernel_1
conv2_bias_1
dense1_kernel_2
dense1_bias_2
dense2_kernel_3
dense2_bias_3
dec_conv1_kernel_4
dec_conv1_bias_4
dec_conv2_kernel_5
dec_conv2_bias_5
dense_2_kernel_6
dense_2_bias_6
dense_3_kernel_8
dense_3_bias_8
dec_conv3_kernel_7
dec_conv3_bias_7
identity

identity_1??conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?!dec_conv1/StatefulPartitionedCall?!dec_conv2/StatefulPartitionedCall?!dec_conv3/StatefulPartitionedCall?dense1/StatefulPartitionedCall?dense2/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCall	input_imgconv1_kernel
conv1_bias*
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
B__inference_conv1_layer_call_and_return_conditional_losses_92950682
conv1/StatefulPartitionedCall?
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0conv2_kernel_1conv2_bias_1*
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
B__inference_conv2_layer_call_and_return_conditional_losses_92950912
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
D__inference_flatten_layer_call_and_return_conditional_losses_92951092
flatten/PartitionedCall?
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_kernel_2dense1_bias_2*
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
C__inference_dense1_layer_call_and_return_conditional_losses_92951282 
dense1/StatefulPartitionedCall?
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_kernel_3dense2_bias_3*
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
C__inference_dense2_layer_call_and_return_conditional_losses_92951512 
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
D__inference_reshape_layer_call_and_return_conditional_losses_92951772
reshape/PartitionedCall?
!dec_conv1/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0dec_conv1_kernel_4dec_conv1_bias_4*
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
F__inference_dec_conv1_layer_call_and_return_conditional_losses_92951962#
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
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_92950182
up_sampling2d/PartitionedCall?
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
J__inference_concatenate_2_layer_call_and_return_conditional_losses_92952212
concatenate_2/PartitionedCall?
!dec_conv2/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0dec_conv2_kernel_5dec_conv2_bias_5*
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
F__inference_dec_conv2_layer_call_and_return_conditional_losses_92952412#
!dec_conv2/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_2_kernel_6dense_2_bias_6*
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
D__inference_dense_2_layer_call_and_return_conditional_losses_92952642!
dense_2/StatefulPartitionedCall?
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
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_92950492!
up_sampling2d_1/PartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_kernel_8dense_3_bias_8*
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
D__inference_dense_3_layer_call_and_return_conditional_losses_92952922!
dense_3/StatefulPartitionedCall?
!dec_conv3/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0dec_conv3_kernel_7dec_conv3_bias_7*
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
F__inference_dec_conv3_layer_call_and_return_conditional_losses_92953152#
!dec_conv3/StatefulPartitionedCall?
IdentityIdentity*dec_conv3/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall"^dec_conv1/StatefulPartitionedCall"^dec_conv2/StatefulPartitionedCall"^dec_conv3/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?

Identity_1Identity(dense_3/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall"^dec_conv1/StatefulPartitionedCall"^dec_conv2/StatefulPartitionedCall"^dec_conv3/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapesx
v:?????????@@:?????????::::::::::::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2F
!dec_conv1/StatefulPartitionedCall!dec_conv1/StatefulPartitionedCall2F
!dec_conv2/StatefulPartitionedCall!dec_conv2/StatefulPartitionedCall2F
!dec_conv3/StatefulPartitionedCall!dec_conv3/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2B
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
?
D__inference_dense_3_layer_call_and_return_conditional_losses_9295292

inputs"
matmul_readvariableop_kernel_8!
biasadd_readvariableop_bias_8
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_kernel_8*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_8*
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
?

?
B__inference_conv1_layer_call_and_return_conditional_losses_9295068

inputs 
conv2d_readvariableop_kernel
biasadd_readvariableop_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_kernel*&
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
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias*
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

?
B__inference_conv1_layer_call_and_return_conditional_losses_9295783

inputs 
conv2d_readvariableop_kernel
biasadd_readvariableop_bias
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_kernel*&
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
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias*
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
?
~
+__inference_dec_conv1_layer_call_fn_9295892

inputs
kernel_4

bias_4
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputskernel_4bias_4*
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
F__inference_dec_conv1_layer_call_and_return_conditional_losses_92951962
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
?D
?
D__inference_model_5_layer_call_and_return_conditional_losses_9295409

inputs
inputs_1
conv1_kernel

conv1_bias
conv2_kernel_1
conv2_bias_1
dense1_kernel_2
dense1_bias_2
dense2_kernel_3
dense2_bias_3
dec_conv1_kernel_4
dec_conv1_bias_4
dec_conv2_kernel_5
dec_conv2_bias_5
dense_2_kernel_6
dense_2_bias_6
dense_3_kernel_8
dense_3_bias_8
dec_conv3_kernel_7
dec_conv3_bias_7
identity

identity_1??conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?!dec_conv1/StatefulPartitionedCall?!dec_conv2/StatefulPartitionedCall?!dec_conv3/StatefulPartitionedCall?dense1/StatefulPartitionedCall?dense2/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_kernel
conv1_bias*
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
B__inference_conv1_layer_call_and_return_conditional_losses_92950682
conv1/StatefulPartitionedCall?
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0conv2_kernel_1conv2_bias_1*
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
B__inference_conv2_layer_call_and_return_conditional_losses_92950912
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
D__inference_flatten_layer_call_and_return_conditional_losses_92951092
flatten/PartitionedCall?
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_kernel_2dense1_bias_2*
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
C__inference_dense1_layer_call_and_return_conditional_losses_92951282 
dense1/StatefulPartitionedCall?
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_kernel_3dense2_bias_3*
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
C__inference_dense2_layer_call_and_return_conditional_losses_92951512 
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
D__inference_reshape_layer_call_and_return_conditional_losses_92951772
reshape/PartitionedCall?
!dec_conv1/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0dec_conv1_kernel_4dec_conv1_bias_4*
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
F__inference_dec_conv1_layer_call_and_return_conditional_losses_92951962#
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
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_92950182
up_sampling2d/PartitionedCall?
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
J__inference_concatenate_2_layer_call_and_return_conditional_losses_92952212
concatenate_2/PartitionedCall?
!dec_conv2/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0dec_conv2_kernel_5dec_conv2_bias_5*
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
F__inference_dec_conv2_layer_call_and_return_conditional_losses_92952412#
!dec_conv2/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_2_kernel_6dense_2_bias_6*
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
D__inference_dense_2_layer_call_and_return_conditional_losses_92952642!
dense_2/StatefulPartitionedCall?
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
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_92950492!
up_sampling2d_1/PartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_kernel_8dense_3_bias_8*
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
D__inference_dense_3_layer_call_and_return_conditional_losses_92952922!
dense_3/StatefulPartitionedCall?
!dec_conv3/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0dec_conv3_kernel_7dec_conv3_bias_7*
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
F__inference_dec_conv3_layer_call_and_return_conditional_losses_92953152#
!dec_conv3/StatefulPartitionedCall?
IdentityIdentity*dec_conv3/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall"^dec_conv1/StatefulPartitionedCall"^dec_conv2/StatefulPartitionedCall"^dec_conv3/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?

Identity_1Identity(dense_3/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall"^dec_conv1/StatefulPartitionedCall"^dec_conv2/StatefulPartitionedCall"^dec_conv3/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapesx
v:?????????@@:?????????::::::::::::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2F
!dec_conv1/StatefulPartitionedCall!dec_conv1/StatefulPartitionedCall2F
!dec_conv2/StatefulPartitionedCall!dec_conv2/StatefulPartitionedCall2F
!dec_conv3/StatefulPartitionedCall!dec_conv3/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2B
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
?
?
F__inference_dec_conv2_layer_call_and_return_conditional_losses_9295241

inputs"
conv2d_readvariableop_kernel_5!
biasadd_readvariableop_bias_5
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_kernel_5*&
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
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_5*
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
?
?
)__inference_model_5_layer_call_fn_9295432
	input_img
	input_act

kernel
bias
kernel_1

bias_1
kernel_2

bias_2
kernel_3

bias_3
kernel_4

bias_4
kernel_5

bias_5
kernel_6

bias_6
kernel_8

bias_8
kernel_7

bias_7
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	input_img	input_actkernelbiaskernel_1bias_1kernel_2bias_2kernel_3bias_3kernel_4bias_4kernel_5bias_5kernel_6bias_6kernel_8bias_8kernel_7bias_7*
Tin
2*
Tout
2*
_collective_manager_ids
 *T
_output_shapesB
@:+???????????????????????????:?????????*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_model_5_layer_call_and_return_conditional_losses_92954092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapesx
v:?????????@@:?????????::::::::::::::::::22
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
C__inference_dense2_layer_call_and_return_conditional_losses_9295848

inputs"
matmul_readvariableop_kernel_3!
biasadd_readvariableop_bias_3
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_kernel_3*
_output_shapes
:	 ?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_3*
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
?
`
D__inference_reshape_layer_call_and_return_conditional_losses_9295869

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
?
E
)__inference_flatten_layer_call_fn_9295819

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
D__inference_flatten_layer_call_and_return_conditional_losses_92951092
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
?
v
J__inference_concatenate_2_layer_call_and_return_conditional_losses_9295917
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
D__inference_dense_3_layer_call_and_return_conditional_losses_9295969

inputs"
matmul_readvariableop_kernel_8!
biasadd_readvariableop_bias_8
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_kernel_8*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_8*
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
?
f
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_9295018

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
F__inference_dec_conv3_layer_call_and_return_conditional_losses_9295315

inputs"
conv2d_readvariableop_kernel_7!
biasadd_readvariableop_bias_7
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_kernel_7*&
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
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_7*
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
?
v
'__inference_conv1_layer_call_fn_9295790

inputs

kernel
bias
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputskernelbias*
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
B__inference_conv1_layer_call_and_return_conditional_losses_92950682
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
D__inference_dense_2_layer_call_and_return_conditional_losses_9295264

inputs"
matmul_readvariableop_kernel_6!
biasadd_readvariableop_bias_6
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_kernel_6*
_output_shapes

:#*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_6*
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
?
{
(__inference_dense1_layer_call_fn_9295837

inputs
kernel_2

bias_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputskernel_2bias_2*
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
C__inference_dense1_layer_call_and_return_conditional_losses_92951282
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
?
t
J__inference_concatenate_2_layer_call_and_return_conditional_losses_9295221

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
?D
?
D__inference_model_5_layer_call_and_return_conditional_losses_9295473

inputs
inputs_1
conv1_kernel

conv1_bias
conv2_kernel_1
conv2_bias_1
dense1_kernel_2
dense1_bias_2
dense2_kernel_3
dense2_bias_3
dec_conv1_kernel_4
dec_conv1_bias_4
dec_conv2_kernel_5
dec_conv2_bias_5
dense_2_kernel_6
dense_2_bias_6
dense_3_kernel_8
dense_3_bias_8
dec_conv3_kernel_7
dec_conv3_bias_7
identity

identity_1??conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?!dec_conv1/StatefulPartitionedCall?!dec_conv2/StatefulPartitionedCall?!dec_conv3/StatefulPartitionedCall?dense1/StatefulPartitionedCall?dense2/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_kernel
conv1_bias*
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
B__inference_conv1_layer_call_and_return_conditional_losses_92950682
conv1/StatefulPartitionedCall?
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0conv2_kernel_1conv2_bias_1*
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
B__inference_conv2_layer_call_and_return_conditional_losses_92950912
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
D__inference_flatten_layer_call_and_return_conditional_losses_92951092
flatten/PartitionedCall?
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_kernel_2dense1_bias_2*
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
C__inference_dense1_layer_call_and_return_conditional_losses_92951282 
dense1/StatefulPartitionedCall?
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_kernel_3dense2_bias_3*
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
C__inference_dense2_layer_call_and_return_conditional_losses_92951512 
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
D__inference_reshape_layer_call_and_return_conditional_losses_92951772
reshape/PartitionedCall?
!dec_conv1/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0dec_conv1_kernel_4dec_conv1_bias_4*
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
F__inference_dec_conv1_layer_call_and_return_conditional_losses_92951962#
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
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_92950182
up_sampling2d/PartitionedCall?
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
J__inference_concatenate_2_layer_call_and_return_conditional_losses_92952212
concatenate_2/PartitionedCall?
!dec_conv2/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0dec_conv2_kernel_5dec_conv2_bias_5*
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
F__inference_dec_conv2_layer_call_and_return_conditional_losses_92952412#
!dec_conv2/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_2_kernel_6dense_2_bias_6*
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
D__inference_dense_2_layer_call_and_return_conditional_losses_92952642!
dense_2/StatefulPartitionedCall?
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
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_92950492!
up_sampling2d_1/PartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_kernel_8dense_3_bias_8*
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
D__inference_dense_3_layer_call_and_return_conditional_losses_92952922!
dense_3/StatefulPartitionedCall?
!dec_conv3/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0dec_conv3_kernel_7dec_conv3_bias_7*
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
F__inference_dec_conv3_layer_call_and_return_conditional_losses_92953152#
!dec_conv3/StatefulPartitionedCall?
IdentityIdentity*dec_conv3/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall"^dec_conv1/StatefulPartitionedCall"^dec_conv2/StatefulPartitionedCall"^dec_conv3/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?

Identity_1Identity(dense_3/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall"^dec_conv1/StatefulPartitionedCall"^dec_conv2/StatefulPartitionedCall"^dec_conv3/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapesx
v:?????????@@:?????????::::::::::::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2F
!dec_conv1/StatefulPartitionedCall!dec_conv1/StatefulPartitionedCall2F
!dec_conv2/StatefulPartitionedCall!dec_conv2/StatefulPartitionedCall2F
!dec_conv3/StatefulPartitionedCall!dec_conv3/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2B
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
?
h
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_9295034

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
z
'__inference_conv2_layer_call_fn_9295808

inputs
kernel_1

bias_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputskernel_1bias_1*
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
B__inference_conv2_layer_call_and_return_conditional_losses_92950912
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
B__inference_conv2_layer_call_and_return_conditional_losses_9295091

inputs"
conv2d_readvariableop_kernel_1!
biasadd_readvariableop_bias_1
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_kernel_1*&
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
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_1*
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
??
?
D__inference_model_5_layer_call_and_return_conditional_losses_9295622
inputs_0
inputs_1&
"conv1_conv2d_readvariableop_kernel%
!conv1_biasadd_readvariableop_bias(
$conv2_conv2d_readvariableop_kernel_1'
#conv2_biasadd_readvariableop_bias_1)
%dense1_matmul_readvariableop_kernel_2(
$dense1_biasadd_readvariableop_bias_2)
%dense2_matmul_readvariableop_kernel_3(
$dense2_biasadd_readvariableop_bias_3,
(dec_conv1_conv2d_readvariableop_kernel_4+
'dec_conv1_biasadd_readvariableop_bias_4,
(dec_conv2_conv2d_readvariableop_kernel_5+
'dec_conv2_biasadd_readvariableop_bias_5*
&dense_2_matmul_readvariableop_kernel_6)
%dense_2_biasadd_readvariableop_bias_6*
&dense_3_matmul_readvariableop_kernel_8)
%dense_3_biasadd_readvariableop_bias_8,
(dec_conv3_conv2d_readvariableop_kernel_7+
'dec_conv3_biasadd_readvariableop_bias_7
identity

identity_1??conv1/BiasAdd/ReadVariableOp?conv1/Conv2D/ReadVariableOp?conv2/BiasAdd/ReadVariableOp?conv2/Conv2D/ReadVariableOp? dec_conv1/BiasAdd/ReadVariableOp?dec_conv1/Conv2D/ReadVariableOp? dec_conv2/BiasAdd/ReadVariableOp?dec_conv2/Conv2D/ReadVariableOp? dec_conv3/BiasAdd/ReadVariableOp?dec_conv3/Conv2D/ReadVariableOp?dense1/BiasAdd/ReadVariableOp?dense1/MatMul/ReadVariableOp?dense2/BiasAdd/ReadVariableOp?dense2/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?
conv1/Conv2D/ReadVariableOpReadVariableOp"conv1_conv2d_readvariableop_kernel*&
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
conv1/BiasAdd/ReadVariableOpReadVariableOp!conv1_biasadd_readvariableop_bias*
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
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_kernel_1*&
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
conv2/BiasAdd/ReadVariableOpReadVariableOp#conv2_biasadd_readvariableop_bias_1*
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
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_kernel_2*
_output_shapes
:	? *
dtype02
dense1/MatMul/ReadVariableOp?
dense1/MatMulMatMulflatten/Reshape:output:0$dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense1/MatMul?
dense1/BiasAdd/ReadVariableOpReadVariableOp$dense1_biasadd_readvariableop_bias_2*
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
dense2/MatMul/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_kernel_3*
_output_shapes
:	 ?*
dtype02
dense2/MatMul/ReadVariableOp?
dense2/MatMulMatMuldense1/Relu:activations:0$dense2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense2/MatMul?
dense2/BiasAdd/ReadVariableOpReadVariableOp$dense2_biasadd_readvariableop_bias_3*
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
dec_conv1/Conv2D/ReadVariableOpReadVariableOp(dec_conv1_conv2d_readvariableop_kernel_4*&
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
 dec_conv1/BiasAdd/ReadVariableOpReadVariableOp'dec_conv1_biasadd_readvariableop_bias_4*
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
*up_sampling2d/resize/ResizeNearestNeighborx
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
dec_conv2/Conv2D/ReadVariableOpReadVariableOp(dec_conv2_conv2d_readvariableop_kernel_5*&
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
 dec_conv2/BiasAdd/ReadVariableOpReadVariableOp'dec_conv2_biasadd_readvariableop_bias_5*
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
dec_conv2/Relu?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_kernel_6*
_output_shapes

:#*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMulconcatenate_2/concat:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp%dense_2_biasadd_readvariableop_bias_6*
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
dense_2/Reluz
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
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_kernel_8*
_output_shapes

:*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp%dense_3_biasadd_readvariableop_bias_8*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/BiasAdd?
dec_conv3/Conv2D/ReadVariableOpReadVariableOp(dec_conv3_conv2d_readvariableop_kernel_7*&
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
 dec_conv3/BiasAdd/ReadVariableOpReadVariableOp'dec_conv3_biasadd_readvariableop_bias_7*
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
dec_conv3/Relu?
IdentityIdentitydec_conv3/Relu:activations:0^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp!^dec_conv1/BiasAdd/ReadVariableOp ^dec_conv1/Conv2D/ReadVariableOp!^dec_conv2/BiasAdd/ReadVariableOp ^dec_conv2/Conv2D/ReadVariableOp!^dec_conv3/BiasAdd/ReadVariableOp ^dec_conv3/Conv2D/ReadVariableOp^dense1/BiasAdd/ReadVariableOp^dense1/MatMul/ReadVariableOp^dense2/BiasAdd/ReadVariableOp^dense2/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????@@2

Identity?

Identity_1Identitydense_3/BiasAdd:output:0^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp!^dec_conv1/BiasAdd/ReadVariableOp ^dec_conv1/Conv2D/ReadVariableOp!^dec_conv2/BiasAdd/ReadVariableOp ^dec_conv2/Conv2D/ReadVariableOp!^dec_conv3/BiasAdd/ReadVariableOp ^dec_conv3/Conv2D/ReadVariableOp^dense1/BiasAdd/ReadVariableOp^dense1/MatMul/ReadVariableOp^dense2/BiasAdd/ReadVariableOp^dense2/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapesx
v:?????????@@:?????????::::::::::::::::::2<
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
dense2/MatMul/ReadVariableOpdense2/MatMul/ReadVariableOp2@
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
F__inference_dec_conv2_layer_call_and_return_conditional_losses_9295903

inputs"
conv2d_readvariableop_kernel_5!
biasadd_readvariableop_bias_5
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_kernel_5*&
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
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_5*
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
K
/__inference_up_sampling2d_layer_call_fn_9295021

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
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_92950182
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
?
{
(__inference_dense2_layer_call_fn_9295855

inputs
kernel_3

bias_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputskernel_3bias_3*
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
C__inference_dense2_layer_call_and_return_conditional_losses_92951512
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
?
[
/__inference_concatenate_2_layer_call_fn_9295923
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
J__inference_concatenate_2_layer_call_and_return_conditional_losses_92952212
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
M
1__inference_up_sampling2d_1_layer_call_fn_9295052

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
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_92950492
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
?
~
+__inference_dec_conv2_layer_call_fn_9295910

inputs
kernel_5

bias_5
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputskernel_5bias_5*
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
F__inference_dec_conv2_layer_call_and_return_conditional_losses_92952412
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
D__inference_dense_2_layer_call_and_return_conditional_losses_9295934

inputs"
matmul_readvariableop_kernel_6!
biasadd_readvariableop_bias_6
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_kernel_6*
_output_shapes

:#*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_6*
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
?
`
D__inference_flatten_layer_call_and_return_conditional_losses_9295814

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
C__inference_dense1_layer_call_and_return_conditional_losses_9295830

inputs"
matmul_readvariableop_kernel_2!
biasadd_readvariableop_bias_2
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_kernel_2*
_output_shapes
:	? *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_2*
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
?
~
+__inference_dec_conv3_layer_call_fn_9295959

inputs
kernel_7

bias_7
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputskernel_7bias_7*
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
F__inference_dec_conv3_layer_call_and_return_conditional_losses_92953152
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
?
)__inference_model_5_layer_call_fn_9295496
	input_img
	input_act

kernel
bias
kernel_1

bias_1
kernel_2

bias_2
kernel_3

bias_3
kernel_4

bias_4
kernel_5

bias_5
kernel_6

bias_6
kernel_8

bias_8
kernel_7

bias_7
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	input_img	input_actkernelbiaskernel_1bias_1kernel_2bias_2kernel_3bias_3kernel_4bias_4kernel_5bias_5kernel_6bias_6kernel_8bias_8kernel_7bias_7*
Tin
2*
Tout
2*
_collective_manager_ids
 *T
_output_shapesB
@:+???????????????????????????:?????????*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_model_5_layer_call_and_return_conditional_losses_92954732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapesx
v:?????????@@:?????????::::::::::::::::::22
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
C__inference_dense2_layer_call_and_return_conditional_losses_9295151

inputs"
matmul_readvariableop_kernel_3!
biasadd_readvariableop_bias_3
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_kernel_3*
_output_shapes
:	 ?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_3*
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
?
f
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_9295003

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
F__inference_dec_conv1_layer_call_and_return_conditional_losses_9295885

inputs"
conv2d_readvariableop_kernel_4!
biasadd_readvariableop_bias_4
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_kernel_4*&
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
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_4*
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
?

?
F__inference_dec_conv1_layer_call_and_return_conditional_losses_9295196

inputs"
conv2d_readvariableop_kernel_4!
biasadd_readvariableop_bias_4
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_kernel_4*&
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
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_4*
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
?
?
)__inference_model_5_layer_call_fn_9295746
inputs_0
inputs_1

kernel
bias
kernel_1

bias_1
kernel_2

bias_2
kernel_3

bias_3
kernel_4

bias_4
kernel_5

bias_5
kernel_6

bias_6
kernel_8

bias_8
kernel_7

bias_7
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1kernelbiaskernel_1bias_1kernel_2bias_2kernel_3bias_3kernel_4bias_4kernel_5bias_5kernel_6bias_6kernel_8bias_8kernel_7bias_7*
Tin
2*
Tout
2*
_collective_manager_ids
 *T
_output_shapesB
@:+???????????????????????????:?????????*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_model_5_layer_call_and_return_conditional_losses_92954092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapesx
v:?????????@@:?????????::::::::::::::::::22
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
?
h
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_9295049

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
?D
?
D__inference_model_5_layer_call_and_return_conditional_losses_9295329
	input_img
	input_act
conv1_kernel

conv1_bias
conv2_kernel_1
conv2_bias_1
dense1_kernel_2
dense1_bias_2
dense2_kernel_3
dense2_bias_3
dec_conv1_kernel_4
dec_conv1_bias_4
dec_conv2_kernel_5
dec_conv2_bias_5
dense_2_kernel_6
dense_2_bias_6
dense_3_kernel_8
dense_3_bias_8
dec_conv3_kernel_7
dec_conv3_bias_7
identity

identity_1??conv1/StatefulPartitionedCall?conv2/StatefulPartitionedCall?!dec_conv1/StatefulPartitionedCall?!dec_conv2/StatefulPartitionedCall?!dec_conv3/StatefulPartitionedCall?dense1/StatefulPartitionedCall?dense2/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?dense_3/StatefulPartitionedCall?
conv1/StatefulPartitionedCallStatefulPartitionedCall	input_imgconv1_kernel
conv1_bias*
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
B__inference_conv1_layer_call_and_return_conditional_losses_92950682
conv1/StatefulPartitionedCall?
conv2/StatefulPartitionedCallStatefulPartitionedCall&conv1/StatefulPartitionedCall:output:0conv2_kernel_1conv2_bias_1*
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
B__inference_conv2_layer_call_and_return_conditional_losses_92950912
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
D__inference_flatten_layer_call_and_return_conditional_losses_92951092
flatten/PartitionedCall?
dense1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense1_kernel_2dense1_bias_2*
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
C__inference_dense1_layer_call_and_return_conditional_losses_92951282 
dense1/StatefulPartitionedCall?
dense2/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0dense2_kernel_3dense2_bias_3*
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
C__inference_dense2_layer_call_and_return_conditional_losses_92951512 
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
D__inference_reshape_layer_call_and_return_conditional_losses_92951772
reshape/PartitionedCall?
!dec_conv1/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0dec_conv1_kernel_4dec_conv1_bias_4*
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
F__inference_dec_conv1_layer_call_and_return_conditional_losses_92951962#
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
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_92950182
up_sampling2d/PartitionedCall?
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
J__inference_concatenate_2_layer_call_and_return_conditional_losses_92952212
concatenate_2/PartitionedCall?
!dec_conv2/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0dec_conv2_kernel_5dec_conv2_bias_5*
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
F__inference_dec_conv2_layer_call_and_return_conditional_losses_92952412#
!dec_conv2/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_2_kernel_6dense_2_bias_6*
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
D__inference_dense_2_layer_call_and_return_conditional_losses_92952642!
dense_2/StatefulPartitionedCall?
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
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_92950492!
up_sampling2d_1/PartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_kernel_8dense_3_bias_8*
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
D__inference_dense_3_layer_call_and_return_conditional_losses_92952922!
dense_3/StatefulPartitionedCall?
!dec_conv3/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0dec_conv3_kernel_7dec_conv3_bias_7*
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
F__inference_dec_conv3_layer_call_and_return_conditional_losses_92953152#
!dec_conv3/StatefulPartitionedCall?
IdentityIdentity*dec_conv3/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall"^dec_conv1/StatefulPartitionedCall"^dec_conv2/StatefulPartitionedCall"^dec_conv3/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?

Identity_1Identity(dense_3/StatefulPartitionedCall:output:0^conv1/StatefulPartitionedCall^conv2/StatefulPartitionedCall"^dec_conv1/StatefulPartitionedCall"^dec_conv2/StatefulPartitionedCall"^dec_conv3/StatefulPartitionedCall^dense1/StatefulPartitionedCall^dense2/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapesx
v:?????????@@:?????????::::::::::::::::::2>
conv1/StatefulPartitionedCallconv1/StatefulPartitionedCall2>
conv2/StatefulPartitionedCallconv2/StatefulPartitionedCall2F
!dec_conv1/StatefulPartitionedCall!dec_conv1/StatefulPartitionedCall2F
!dec_conv2/StatefulPartitionedCall!dec_conv2/StatefulPartitionedCall2F
!dec_conv3/StatefulPartitionedCall!dec_conv3/StatefulPartitionedCall2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2@
dense2/StatefulPartitionedCalldense2/StatefulPartitionedCall2B
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
)__inference_model_5_layer_call_fn_9295772
inputs_0
inputs_1

kernel
bias
kernel_1

bias_1
kernel_2

bias_2
kernel_3

bias_3
kernel_4

bias_4
kernel_5

bias_5
kernel_6

bias_6
kernel_8

bias_8
kernel_7

bias_7
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1kernelbiaskernel_1bias_1kernel_2bias_2kernel_3bias_3kernel_4bias_4kernel_5bias_5kernel_6bias_6kernel_8bias_8kernel_7bias_7*
Tin
2*
Tout
2*
_collective_manager_ids
 *T
_output_shapesB
@:+???????????????????????????:?????????*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8? *M
fHRF
D__inference_model_5_layer_call_and_return_conditional_losses_92954732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapesx
v:?????????@@:?????????::::::::::::::::::22
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
?
?
%__inference_signature_wrapper_9295524
	input_act
	input_img

kernel
bias
kernel_1

bias_1
kernel_2

bias_2
kernel_3

bias_3
kernel_4

bias_4
kernel_5

bias_5
kernel_6

bias_6
kernel_8

bias_8
kernel_7

bias_7
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall	input_img	input_actkernelbiaskernel_1bias_1kernel_2bias_2kernel_3bias_3kernel_4bias_4kernel_5bias_5kernel_6bias_6kernel_8bias_8kernel_7bias_7*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:?????????@@:?????????*4
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8? *+
f&R$
"__inference__wrapped_model_92949902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@@2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapesx
v:?????????:?????????@@::::::::::::::::::22
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
??
?
D__inference_model_5_layer_call_and_return_conditional_losses_9295720
inputs_0
inputs_1&
"conv1_conv2d_readvariableop_kernel%
!conv1_biasadd_readvariableop_bias(
$conv2_conv2d_readvariableop_kernel_1'
#conv2_biasadd_readvariableop_bias_1)
%dense1_matmul_readvariableop_kernel_2(
$dense1_biasadd_readvariableop_bias_2)
%dense2_matmul_readvariableop_kernel_3(
$dense2_biasadd_readvariableop_bias_3,
(dec_conv1_conv2d_readvariableop_kernel_4+
'dec_conv1_biasadd_readvariableop_bias_4,
(dec_conv2_conv2d_readvariableop_kernel_5+
'dec_conv2_biasadd_readvariableop_bias_5*
&dense_2_matmul_readvariableop_kernel_6)
%dense_2_biasadd_readvariableop_bias_6*
&dense_3_matmul_readvariableop_kernel_8)
%dense_3_biasadd_readvariableop_bias_8,
(dec_conv3_conv2d_readvariableop_kernel_7+
'dec_conv3_biasadd_readvariableop_bias_7
identity

identity_1??conv1/BiasAdd/ReadVariableOp?conv1/Conv2D/ReadVariableOp?conv2/BiasAdd/ReadVariableOp?conv2/Conv2D/ReadVariableOp? dec_conv1/BiasAdd/ReadVariableOp?dec_conv1/Conv2D/ReadVariableOp? dec_conv2/BiasAdd/ReadVariableOp?dec_conv2/Conv2D/ReadVariableOp? dec_conv3/BiasAdd/ReadVariableOp?dec_conv3/Conv2D/ReadVariableOp?dense1/BiasAdd/ReadVariableOp?dense1/MatMul/ReadVariableOp?dense2/BiasAdd/ReadVariableOp?dense2/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?
conv1/Conv2D/ReadVariableOpReadVariableOp"conv1_conv2d_readvariableop_kernel*&
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
conv1/BiasAdd/ReadVariableOpReadVariableOp!conv1_biasadd_readvariableop_bias*
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
conv2/Conv2D/ReadVariableOpReadVariableOp$conv2_conv2d_readvariableop_kernel_1*&
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
conv2/BiasAdd/ReadVariableOpReadVariableOp#conv2_biasadd_readvariableop_bias_1*
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
dense1/MatMul/ReadVariableOpReadVariableOp%dense1_matmul_readvariableop_kernel_2*
_output_shapes
:	? *
dtype02
dense1/MatMul/ReadVariableOp?
dense1/MatMulMatMulflatten/Reshape:output:0$dense1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:????????? 2
dense1/MatMul?
dense1/BiasAdd/ReadVariableOpReadVariableOp$dense1_biasadd_readvariableop_bias_2*
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
dense2/MatMul/ReadVariableOpReadVariableOp%dense2_matmul_readvariableop_kernel_3*
_output_shapes
:	 ?*
dtype02
dense2/MatMul/ReadVariableOp?
dense2/MatMulMatMuldense1/Relu:activations:0$dense2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense2/MatMul?
dense2/BiasAdd/ReadVariableOpReadVariableOp$dense2_biasadd_readvariableop_bias_3*
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
dec_conv1/Conv2D/ReadVariableOpReadVariableOp(dec_conv1_conv2d_readvariableop_kernel_4*&
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
 dec_conv1/BiasAdd/ReadVariableOpReadVariableOp'dec_conv1_biasadd_readvariableop_bias_4*
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
*up_sampling2d/resize/ResizeNearestNeighborx
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
dec_conv2/Conv2D/ReadVariableOpReadVariableOp(dec_conv2_conv2d_readvariableop_kernel_5*&
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
 dec_conv2/BiasAdd/ReadVariableOpReadVariableOp'dec_conv2_biasadd_readvariableop_bias_5*
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
dec_conv2/Relu?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_kernel_6*
_output_shapes

:#*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMulconcatenate_2/concat:output:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp%dense_2_biasadd_readvariableop_bias_6*
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
dense_2/Reluz
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
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_kernel_8*
_output_shapes

:*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp%dense_3_biasadd_readvariableop_bias_8*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/BiasAdd?
dec_conv3/Conv2D/ReadVariableOpReadVariableOp(dec_conv3_conv2d_readvariableop_kernel_7*&
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
 dec_conv3/BiasAdd/ReadVariableOpReadVariableOp'dec_conv3_biasadd_readvariableop_bias_7*
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
dec_conv3/Relu?
IdentityIdentitydec_conv3/Relu:activations:0^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp!^dec_conv1/BiasAdd/ReadVariableOp ^dec_conv1/Conv2D/ReadVariableOp!^dec_conv2/BiasAdd/ReadVariableOp ^dec_conv2/Conv2D/ReadVariableOp!^dec_conv3/BiasAdd/ReadVariableOp ^dec_conv3/Conv2D/ReadVariableOp^dense1/BiasAdd/ReadVariableOp^dense1/MatMul/ReadVariableOp^dense2/BiasAdd/ReadVariableOp^dense2/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????@@2

Identity?

Identity_1Identitydense_3/BiasAdd:output:0^conv1/BiasAdd/ReadVariableOp^conv1/Conv2D/ReadVariableOp^conv2/BiasAdd/ReadVariableOp^conv2/Conv2D/ReadVariableOp!^dec_conv1/BiasAdd/ReadVariableOp ^dec_conv1/Conv2D/ReadVariableOp!^dec_conv2/BiasAdd/ReadVariableOp ^dec_conv2/Conv2D/ReadVariableOp!^dec_conv3/BiasAdd/ReadVariableOp ^dec_conv3/Conv2D/ReadVariableOp^dense1/BiasAdd/ReadVariableOp^dense1/MatMul/ReadVariableOp^dense2/BiasAdd/ReadVariableOp^dense2/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*?
_input_shapesx
v:?????????@@:?????????::::::::::::::::::2<
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
dense2/MatMul/ReadVariableOpdense2/MatMul/ReadVariableOp2@
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
?
|
)__inference_dense_3_layer_call_fn_9295976

inputs
kernel_8

bias_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputskernel_8bias_8*
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
D__inference_dense_3_layer_call_and_return_conditional_losses_92952922
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
?
|
)__inference_dense_2_layer_call_fn_9295941

inputs
kernel_6

bias_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputskernel_6bias_6*
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
D__inference_dense_2_layer_call_and_return_conditional_losses_92952642
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
?
`
D__inference_flatten_layer_call_and_return_conditional_losses_9295109

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
??
?!
#__inference__traced_restore_9296365
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
"assignvariableop_11_dec_conv2_bias&
"assignvariableop_12_dense_2_kernel$
 assignvariableop_13_dense_2_bias(
$assignvariableop_14_dec_conv3_kernel&
"assignvariableop_15_dec_conv3_bias&
"assignvariableop_16_dense_3_kernel$
 assignvariableop_17_dense_3_bias*
&assignvariableop_18_training_adam_iter,
(assignvariableop_19_training_adam_beta_1,
(assignvariableop_20_training_adam_beta_2+
'assignvariableop_21_training_adam_decay3
/assignvariableop_22_training_adam_learning_rate4
0assignvariableop_23_training_adam_conv1_kernel_m2
.assignvariableop_24_training_adam_conv1_bias_m4
0assignvariableop_25_training_adam_conv2_kernel_m2
.assignvariableop_26_training_adam_conv2_bias_m5
1assignvariableop_27_training_adam_dense1_kernel_m3
/assignvariableop_28_training_adam_dense1_bias_m5
1assignvariableop_29_training_adam_dense2_kernel_m3
/assignvariableop_30_training_adam_dense2_bias_m8
4assignvariableop_31_training_adam_dec_conv1_kernel_m6
2assignvariableop_32_training_adam_dec_conv1_bias_m8
4assignvariableop_33_training_adam_dec_conv2_kernel_m6
2assignvariableop_34_training_adam_dec_conv2_bias_m6
2assignvariableop_35_training_adam_dense_2_kernel_m4
0assignvariableop_36_training_adam_dense_2_bias_m8
4assignvariableop_37_training_adam_dec_conv3_kernel_m6
2assignvariableop_38_training_adam_dec_conv3_bias_m6
2assignvariableop_39_training_adam_dense_3_kernel_m4
0assignvariableop_40_training_adam_dense_3_bias_m4
0assignvariableop_41_training_adam_conv1_kernel_v2
.assignvariableop_42_training_adam_conv1_bias_v4
0assignvariableop_43_training_adam_conv2_kernel_v2
.assignvariableop_44_training_adam_conv2_bias_v5
1assignvariableop_45_training_adam_dense1_kernel_v3
/assignvariableop_46_training_adam_dense1_bias_v5
1assignvariableop_47_training_adam_dense2_kernel_v3
/assignvariableop_48_training_adam_dense2_bias_v8
4assignvariableop_49_training_adam_dec_conv1_kernel_v6
2assignvariableop_50_training_adam_dec_conv1_bias_v8
4assignvariableop_51_training_adam_dec_conv2_kernel_v6
2assignvariableop_52_training_adam_dec_conv2_bias_v6
2assignvariableop_53_training_adam_dense_2_kernel_v4
0assignvariableop_54_training_adam_dense_2_bias_v8
4assignvariableop_55_training_adam_dec_conv3_kernel_v6
2assignvariableop_56_training_adam_dec_conv3_bias_v6
2assignvariableop_57_training_adam_dense_3_kernel_v4
0assignvariableop_58_training_adam_dense_3_bias_v
identity_60??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?"
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:<*
dtype0*?!
value?!B?!<B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:<*
dtype0*?
value?B?<B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*J
dtypes@
>2<	2
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
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_2_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp$assignvariableop_14_dec_conv3_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp"assignvariableop_15_dec_conv3_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_3_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp assignvariableop_17_dense_3_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp&assignvariableop_18_training_adam_iterIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp(assignvariableop_19_training_adam_beta_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp(assignvariableop_20_training_adam_beta_2Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp'assignvariableop_21_training_adam_decayIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp/assignvariableop_22_training_adam_learning_rateIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp0assignvariableop_23_training_adam_conv1_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp.assignvariableop_24_training_adam_conv1_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp0assignvariableop_25_training_adam_conv2_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp.assignvariableop_26_training_adam_conv2_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp1assignvariableop_27_training_adam_dense1_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp/assignvariableop_28_training_adam_dense1_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp1assignvariableop_29_training_adam_dense2_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp/assignvariableop_30_training_adam_dense2_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp4assignvariableop_31_training_adam_dec_conv1_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp2assignvariableop_32_training_adam_dec_conv1_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp4assignvariableop_33_training_adam_dec_conv2_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp2assignvariableop_34_training_adam_dec_conv2_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp2assignvariableop_35_training_adam_dense_2_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp0assignvariableop_36_training_adam_dense_2_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp4assignvariableop_37_training_adam_dec_conv3_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp2assignvariableop_38_training_adam_dec_conv3_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp2assignvariableop_39_training_adam_dense_3_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp0assignvariableop_40_training_adam_dense_3_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp0assignvariableop_41_training_adam_conv1_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp.assignvariableop_42_training_adam_conv1_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp0assignvariableop_43_training_adam_conv2_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp.assignvariableop_44_training_adam_conv2_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp1assignvariableop_45_training_adam_dense1_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp/assignvariableop_46_training_adam_dense1_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp1assignvariableop_47_training_adam_dense2_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp/assignvariableop_48_training_adam_dense2_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp4assignvariableop_49_training_adam_dec_conv1_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp2assignvariableop_50_training_adam_dec_conv1_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp4assignvariableop_51_training_adam_dec_conv2_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp2assignvariableop_52_training_adam_dec_conv2_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp2assignvariableop_53_training_adam_dense_2_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp0assignvariableop_54_training_adam_dense_2_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp4assignvariableop_55_training_adam_dec_conv3_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp2assignvariableop_56_training_adam_dec_conv3_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp2assignvariableop_57_training_adam_dense_3_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp0assignvariableop_58_training_adam_dense_3_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_589
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_59Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_59?

Identity_60IdentityIdentity_59:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_60"#
identity_60Identity_60:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582(
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
F__inference_dec_conv3_layer_call_and_return_conditional_losses_9295952

inputs"
conv2d_readvariableop_kernel_7!
biasadd_readvariableop_bias_7
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_kernel_7*&
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
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_bias_7*
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
?}
?
 __inference__traced_save_9296178
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
)savev2_dec_conv2_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop/
+savev2_dec_conv3_kernel_read_readvariableop-
)savev2_dec_conv3_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop1
-savev2_training_adam_iter_read_readvariableop	3
/savev2_training_adam_beta_1_read_readvariableop3
/savev2_training_adam_beta_2_read_readvariableop2
.savev2_training_adam_decay_read_readvariableop:
6savev2_training_adam_learning_rate_read_readvariableop;
7savev2_training_adam_conv1_kernel_m_read_readvariableop9
5savev2_training_adam_conv1_bias_m_read_readvariableop;
7savev2_training_adam_conv2_kernel_m_read_readvariableop9
5savev2_training_adam_conv2_bias_m_read_readvariableop<
8savev2_training_adam_dense1_kernel_m_read_readvariableop:
6savev2_training_adam_dense1_bias_m_read_readvariableop<
8savev2_training_adam_dense2_kernel_m_read_readvariableop:
6savev2_training_adam_dense2_bias_m_read_readvariableop?
;savev2_training_adam_dec_conv1_kernel_m_read_readvariableop=
9savev2_training_adam_dec_conv1_bias_m_read_readvariableop?
;savev2_training_adam_dec_conv2_kernel_m_read_readvariableop=
9savev2_training_adam_dec_conv2_bias_m_read_readvariableop=
9savev2_training_adam_dense_2_kernel_m_read_readvariableop;
7savev2_training_adam_dense_2_bias_m_read_readvariableop?
;savev2_training_adam_dec_conv3_kernel_m_read_readvariableop=
9savev2_training_adam_dec_conv3_bias_m_read_readvariableop=
9savev2_training_adam_dense_3_kernel_m_read_readvariableop;
7savev2_training_adam_dense_3_bias_m_read_readvariableop;
7savev2_training_adam_conv1_kernel_v_read_readvariableop9
5savev2_training_adam_conv1_bias_v_read_readvariableop;
7savev2_training_adam_conv2_kernel_v_read_readvariableop9
5savev2_training_adam_conv2_bias_v_read_readvariableop<
8savev2_training_adam_dense1_kernel_v_read_readvariableop:
6savev2_training_adam_dense1_bias_v_read_readvariableop<
8savev2_training_adam_dense2_kernel_v_read_readvariableop:
6savev2_training_adam_dense2_bias_v_read_readvariableop?
;savev2_training_adam_dec_conv1_kernel_v_read_readvariableop=
9savev2_training_adam_dec_conv1_bias_v_read_readvariableop?
;savev2_training_adam_dec_conv2_kernel_v_read_readvariableop=
9savev2_training_adam_dec_conv2_bias_v_read_readvariableop=
9savev2_training_adam_dense_2_kernel_v_read_readvariableop;
7savev2_training_adam_dense_2_bias_v_read_readvariableop?
;savev2_training_adam_dec_conv3_kernel_v_read_readvariableop=
9savev2_training_adam_dec_conv3_bias_v_read_readvariableop=
9savev2_training_adam_dense_3_kernel_v_read_readvariableop;
7savev2_training_adam_dense_3_bias_v_read_readvariableop
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
ShardedFilename?"
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:<*
dtype0*?!
value?!B?!<B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:<*
dtype0*?
value?B?<B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_conv1_kernel_read_readvariableop%savev2_conv1_bias_read_readvariableop'savev2_conv2_kernel_read_readvariableop%savev2_conv2_bias_read_readvariableop(savev2_dense1_kernel_read_readvariableop&savev2_dense1_bias_read_readvariableop(savev2_dense2_kernel_read_readvariableop&savev2_dense2_bias_read_readvariableop+savev2_dec_conv1_kernel_read_readvariableop)savev2_dec_conv1_bias_read_readvariableop+savev2_dec_conv2_kernel_read_readvariableop)savev2_dec_conv2_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop+savev2_dec_conv3_kernel_read_readvariableop)savev2_dec_conv3_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop-savev2_training_adam_iter_read_readvariableop/savev2_training_adam_beta_1_read_readvariableop/savev2_training_adam_beta_2_read_readvariableop.savev2_training_adam_decay_read_readvariableop6savev2_training_adam_learning_rate_read_readvariableop7savev2_training_adam_conv1_kernel_m_read_readvariableop5savev2_training_adam_conv1_bias_m_read_readvariableop7savev2_training_adam_conv2_kernel_m_read_readvariableop5savev2_training_adam_conv2_bias_m_read_readvariableop8savev2_training_adam_dense1_kernel_m_read_readvariableop6savev2_training_adam_dense1_bias_m_read_readvariableop8savev2_training_adam_dense2_kernel_m_read_readvariableop6savev2_training_adam_dense2_bias_m_read_readvariableop;savev2_training_adam_dec_conv1_kernel_m_read_readvariableop9savev2_training_adam_dec_conv1_bias_m_read_readvariableop;savev2_training_adam_dec_conv2_kernel_m_read_readvariableop9savev2_training_adam_dec_conv2_bias_m_read_readvariableop9savev2_training_adam_dense_2_kernel_m_read_readvariableop7savev2_training_adam_dense_2_bias_m_read_readvariableop;savev2_training_adam_dec_conv3_kernel_m_read_readvariableop9savev2_training_adam_dec_conv3_bias_m_read_readvariableop9savev2_training_adam_dense_3_kernel_m_read_readvariableop7savev2_training_adam_dense_3_bias_m_read_readvariableop7savev2_training_adam_conv1_kernel_v_read_readvariableop5savev2_training_adam_conv1_bias_v_read_readvariableop7savev2_training_adam_conv2_kernel_v_read_readvariableop5savev2_training_adam_conv2_bias_v_read_readvariableop8savev2_training_adam_dense1_kernel_v_read_readvariableop6savev2_training_adam_dense1_bias_v_read_readvariableop8savev2_training_adam_dense2_kernel_v_read_readvariableop6savev2_training_adam_dense2_bias_v_read_readvariableop;savev2_training_adam_dec_conv1_kernel_v_read_readvariableop9savev2_training_adam_dec_conv1_bias_v_read_readvariableop;savev2_training_adam_dec_conv2_kernel_v_read_readvariableop9savev2_training_adam_dec_conv2_bias_v_read_readvariableop9savev2_training_adam_dense_2_kernel_v_read_readvariableop7savev2_training_adam_dense_2_bias_v_read_readvariableop;savev2_training_adam_dec_conv3_kernel_v_read_readvariableop9savev2_training_adam_dec_conv3_bias_v_read_readvariableop9savev2_training_adam_dense_3_kernel_v_read_readvariableop7savev2_training_adam_dense_3_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *J
dtypes@
>2<	2
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :::::	? : :	 ?:?:::
::#:::::: : : : : :::::	? : :	 ?:?:::
::#::::::::::	? : :	 ?:?:::
::#:::::: 2(
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
::$ 

_output_shapes

:#: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	? : 

_output_shapes
: :%!

_output_shapes
:	 ?:!

_output_shapes	
:?:, (
&
_output_shapes
:: !

_output_shapes
::,"(
&
_output_shapes
:
: #

_output_shapes
::$$ 

_output_shapes

:#: %

_output_shapes
::,&(
&
_output_shapes
:: '

_output_shapes
::$( 

_output_shapes

:: )

_output_shapes
::,*(
&
_output_shapes
:: +

_output_shapes
::,,(
&
_output_shapes
:: -

_output_shapes
::%.!

_output_shapes
:	? : /

_output_shapes
: :%0!

_output_shapes
:	 ?:!1

_output_shapes	
:?:,2(
&
_output_shapes
:: 3

_output_shapes
::,4(
&
_output_shapes
:
: 5

_output_shapes
::$6 

_output_shapes

:#: 7

_output_shapes
::,8(
&
_output_shapes
:: 9

_output_shapes
::$: 

_output_shapes

:: ;

_output_shapes
::<

_output_shapes
: "?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
?
	input_act2
serving_default_input_act:0?????????
G
	input_img:
serving_default_input_img:0?????????@@E
	dec_conv38
StatefulPartitionedCall:0?????????@@;
dense_30
StatefulPartitionedCall:1?????????tensorflow/serving/predict:??
??
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

layer-9
layer_with_weights-5
layer-10
layer-11
layer-12
layer_with_weights-6
layer-13
layer_with_weights-7
layer-14
layer_with_weights-8
layer-15
	optimizer

signatures
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"?~
_tf_keras_network?~{"class_name": "Functional", "name": "model_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_img"}, "name": "input_img", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [8, 8]}, "strides": {"class_name": "__tuple__", "items": [4, 4]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1", "inbound_nodes": [[["input_img", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2", "inbound_nodes": [[["conv1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["conv2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense1", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense2", "trainable": true, "dtype": "float32", "units": 1536, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense2", "inbound_nodes": [[["dense1", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [12, 8, 16]}}, "name": "reshape", "inbound_nodes": [[["dense2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "dec_conv1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_conv1", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d", "inbound_nodes": [[["dec_conv1", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_act"}, "name": "input_act", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "dec_conv2", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [10, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_conv2", "inbound_nodes": [[["up_sampling2d", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_2", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_2", "inbound_nodes": [[["dense1", 0, 0, {}], ["input_act", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [5, 5]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_1", "inbound_nodes": [[["dec_conv2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["concatenate_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "dec_conv3", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [12, 12]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_conv3", "inbound_nodes": [[["up_sampling2d_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}], "input_layers": [["input_img", 0, 0], ["input_act", 0, 0]], "output_layers": [["dec_conv3", 0, 0], ["dense_3", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 3]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 64, 64, 3]}, {"class_name": "TensorShape", "items": [null, 3]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_img"}, "name": "input_img", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [8, 8]}, "strides": {"class_name": "__tuple__", "items": [4, 4]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1", "inbound_nodes": [[["input_img", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2", "inbound_nodes": [[["conv1", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["conv2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense1", "inbound_nodes": [[["flatten", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense2", "trainable": true, "dtype": "float32", "units": 1536, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense2", "inbound_nodes": [[["dense1", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [12, 8, 16]}}, "name": "reshape", "inbound_nodes": [[["dense2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "dec_conv1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_conv1", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d", "inbound_nodes": [[["dec_conv1", 0, 0, {}]]]}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_act"}, "name": "input_act", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "dec_conv2", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [10, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_conv2", "inbound_nodes": [[["up_sampling2d", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_2", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_2", "inbound_nodes": [[["dense1", 0, 0, {}], ["input_act", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [5, 5]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_1", "inbound_nodes": [[["dec_conv2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["concatenate_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "dec_conv3", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [12, 12]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dec_conv3", "inbound_nodes": [[["up_sampling2d_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}], "input_layers": [["input_img", 0, 0], ["input_act", 0, 0]], "output_layers": [["dec_conv3", 0, 0], ["dense_3", 0, 0]]}}, "training_config": {"loss": ["mean_squared_error", "mean_squared_error"], "metrics": [], "loss_weights": [0.1, 1.0], "sample_weight_mode": null, "weighted_metrics": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
#_self_saveable_object_factories"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_img", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 64, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_img"}}
?


kernel
bias
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [8, 8]}, "strides": {"class_name": "__tuple__", "items": [4, 4]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 3]}}
?


 kernel
!bias
#"_self_saveable_object_factories
#regularization_losses
$	variables
%trainable_variables
&	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 15, 15, 8]}}
?
#'_self_saveable_object_factories
(regularization_losses
)	variables
*trainable_variables
+	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

,kernel
-bias
#._self_saveable_object_factories
/regularization_losses
0	variables
1trainable_variables
2	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2304}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2304]}}
?

3kernel
4bias
#5_self_saveable_object_factories
6regularization_losses
7	variables
8trainable_variables
9	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense2", "trainable": true, "dtype": "float32", "units": 1536, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
?
#:_self_saveable_object_factories
;regularization_losses
<	variables
=trainable_variables
>	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [12, 8, 16]}}}
?


?kernel
@bias
#A_self_saveable_object_factories
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "dec_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dec_conv1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 8, 16]}}
?
#F_self_saveable_object_factories
Gregularization_losses
H	variables
Itrainable_variables
J	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "UpSampling2D", "name": "up_sampling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
#K_self_saveable_object_factories"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_act", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_act"}}
?


Lkernel
Mbias
#N_self_saveable_object_factories
Oregularization_losses
P	variables
Qtrainable_variables
R	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "dec_conv2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dec_conv2", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [10, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 16, 16]}}
?
#S_self_saveable_object_factories
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_2", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 32]}, {"class_name": "TensorShape", "items": [null, 3]}]}
?
#X_self_saveable_object_factories
Yregularization_losses
Z	variables
[trainable_variables
\	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "UpSampling2D", "name": "up_sampling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [5, 5]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?

]kernel
^bias
#__self_saveable_object_factories
`regularization_losses
a	variables
btrainable_variables
c	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 20, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 35}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 35]}}
?


dkernel
ebias
#f_self_saveable_object_factories
gregularization_losses
h	variables
itrainable_variables
j	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "dec_conv3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dec_conv3", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [12, 12]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 75, 75, 8]}}
?

kkernel
lbias
#m_self_saveable_object_factories
nregularization_losses
o	variables
ptrainable_variables
q	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 20}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 20]}}
?
riter

sbeta_1

tbeta_2
	udecay
vlearning_ratem?m? m?!m?,m?-m?3m?4m??m?@m?Lm?Mm?]m?^m?dm?em?km?lm?v?v? v?!v?,v?-v?3v?4v??v?@v?Lv?Mv?]v?^v?dv?ev?kv?lv?"
	optimizer
-
?serving_default"
signature_map
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
?
0
1
 2
!3
,4
-5
36
47
?8
@9
L10
M11
]12
^13
d14
e15
k16
l17"
trackable_list_wrapper
?
0
1
 2
!3
,4
-5
36
47
?8
@9
L10
M11
]12
^13
d14
e15
k16
l17"
trackable_list_wrapper
?
wlayer_metrics
regularization_losses
xlayer_regularization_losses
ynon_trainable_variables

zlayers
	variables
{metrics
trainable_variables
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
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
|layer_metrics
}layer_regularization_losses
regularization_losses
~non_trainable_variables

layers
	variables
?metrics
trainable_variables
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
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
#regularization_losses
?non_trainable_variables
?layers
$	variables
?metrics
%trainable_variables
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
(regularization_losses
?non_trainable_variables
?layers
)	variables
?metrics
*trainable_variables
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
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
/regularization_losses
?non_trainable_variables
?layers
0	variables
?metrics
1trainable_variables
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
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
6regularization_losses
?non_trainable_variables
?layers
7	variables
?metrics
8trainable_variables
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
;regularization_losses
?non_trainable_variables
?layers
<	variables
?metrics
=trainable_variables
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
?0
@1"
trackable_list_wrapper
.
?0
@1"
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
Gregularization_losses
?non_trainable_variables
?layers
H	variables
?metrics
Itrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
*:(
2dec_conv2/kernel
:2dec_conv2/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
Oregularization_losses
?non_trainable_variables
?layers
P	variables
?metrics
Qtrainable_variables
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
Tregularization_losses
?non_trainable_variables
?layers
U	variables
?metrics
Vtrainable_variables
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
Yregularization_losses
?non_trainable_variables
?layers
Z	variables
?metrics
[trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :#2dense_2/kernel
:2dense_2/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
`regularization_losses
?non_trainable_variables
?layers
a	variables
?metrics
btrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
*:(2dec_conv3/kernel
:2dec_conv3/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
gregularization_losses
?non_trainable_variables
?layers
h	variables
?metrics
itrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :2dense_3/kernel
:2dense_3/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
?
?layer_metrics
 ?layer_regularization_losses
nregularization_losses
?non_trainable_variables
?layers
o	variables
?metrics
ptrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2training/Adam/iter
: (2training/Adam/beta_1
: (2training/Adam/beta_2
: (2training/Adam/decay
%:# (2training/Adam/learning_rate
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
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
12
13
14
15"
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
4:22training/Adam/conv1/kernel/m
&:$2training/Adam/conv1/bias/m
4:22training/Adam/conv2/kernel/m
&:$2training/Adam/conv2/bias/m
.:,	? 2training/Adam/dense1/kernel/m
':% 2training/Adam/dense1/bias/m
.:,	 ?2training/Adam/dense2/kernel/m
(:&?2training/Adam/dense2/bias/m
8:62 training/Adam/dec_conv1/kernel/m
*:(2training/Adam/dec_conv1/bias/m
8:6
2 training/Adam/dec_conv2/kernel/m
*:(2training/Adam/dec_conv2/bias/m
.:,#2training/Adam/dense_2/kernel/m
(:&2training/Adam/dense_2/bias/m
8:62 training/Adam/dec_conv3/kernel/m
*:(2training/Adam/dec_conv3/bias/m
.:,2training/Adam/dense_3/kernel/m
(:&2training/Adam/dense_3/bias/m
4:22training/Adam/conv1/kernel/v
&:$2training/Adam/conv1/bias/v
4:22training/Adam/conv2/kernel/v
&:$2training/Adam/conv2/bias/v
.:,	? 2training/Adam/dense1/kernel/v
':% 2training/Adam/dense1/bias/v
.:,	 ?2training/Adam/dense2/kernel/v
(:&?2training/Adam/dense2/bias/v
8:62 training/Adam/dec_conv1/kernel/v
*:(2training/Adam/dec_conv1/bias/v
8:6
2 training/Adam/dec_conv2/kernel/v
*:(2training/Adam/dec_conv2/bias/v
.:,#2training/Adam/dense_2/kernel/v
(:&2training/Adam/dense_2/bias/v
8:62 training/Adam/dec_conv3/kernel/v
*:(2training/Adam/dec_conv3/bias/v
.:,2training/Adam/dense_3/kernel/v
(:&2training/Adam/dense_3/bias/v
?2?
)__inference_model_5_layer_call_fn_9295432
)__inference_model_5_layer_call_fn_9295746
)__inference_model_5_layer_call_fn_9295496
)__inference_model_5_layer_call_fn_9295772?
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
D__inference_model_5_layer_call_and_return_conditional_losses_9295720
D__inference_model_5_layer_call_and_return_conditional_losses_9295622
D__inference_model_5_layer_call_and_return_conditional_losses_9295329
D__inference_model_5_layer_call_and_return_conditional_losses_9295367?
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
"__inference__wrapped_model_9294990?
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
'__inference_conv1_layer_call_fn_9295790?
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
B__inference_conv1_layer_call_and_return_conditional_losses_9295783?
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
'__inference_conv2_layer_call_fn_9295808?
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
B__inference_conv2_layer_call_and_return_conditional_losses_9295801?
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
)__inference_flatten_layer_call_fn_9295819?
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
D__inference_flatten_layer_call_and_return_conditional_losses_9295814?
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
(__inference_dense1_layer_call_fn_9295837?
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
C__inference_dense1_layer_call_and_return_conditional_losses_9295830?
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
(__inference_dense2_layer_call_fn_9295855?
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
C__inference_dense2_layer_call_and_return_conditional_losses_9295848?
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
)__inference_reshape_layer_call_fn_9295874?
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
D__inference_reshape_layer_call_and_return_conditional_losses_9295869?
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
+__inference_dec_conv1_layer_call_fn_9295892?
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
F__inference_dec_conv1_layer_call_and_return_conditional_losses_9295885?
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
/__inference_up_sampling2d_layer_call_fn_9295021?
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
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_9295003?
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
+__inference_dec_conv2_layer_call_fn_9295910?
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
F__inference_dec_conv2_layer_call_and_return_conditional_losses_9295903?
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
/__inference_concatenate_2_layer_call_fn_9295923?
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
J__inference_concatenate_2_layer_call_and_return_conditional_losses_9295917?
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
1__inference_up_sampling2d_1_layer_call_fn_9295052?
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
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_9295034?
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
)__inference_dense_2_layer_call_fn_9295941?
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
D__inference_dense_2_layer_call_and_return_conditional_losses_9295934?
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
+__inference_dec_conv3_layer_call_fn_9295959?
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
F__inference_dec_conv3_layer_call_and_return_conditional_losses_9295952?
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
)__inference_dense_3_layer_call_fn_9295976?
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
D__inference_dense_3_layer_call_and_return_conditional_losses_9295969?
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
%__inference_signature_wrapper_9295524	input_act	input_img"?
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
 ?
"__inference__wrapped_model_9294990? !,-34?@LM]^klded?a
Z?W
U?R
+?(
	input_img?????????@@
#? 
	input_act?????????
? "k?h
8
	dec_conv3+?(
	dec_conv3?????????@@
,
dense_3!?
dense_3??????????
J__inference_concatenate_2_layer_call_and_return_conditional_losses_9295917?Z?W
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
/__inference_concatenate_2_layer_call_fn_9295923vZ?W
P?M
K?H
"?
inputs/0????????? 
"?
inputs/1?????????
? "??????????#?
B__inference_conv1_layer_call_and_return_conditional_losses_9295783l7?4
-?*
(?%
inputs?????????@@
? "-?*
#? 
0?????????
? ?
'__inference_conv1_layer_call_fn_9295790_7?4
-?*
(?%
inputs?????????@@
? " ???????????
B__inference_conv2_layer_call_and_return_conditional_losses_9295801l !7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
'__inference_conv2_layer_call_fn_9295808_ !7?4
-?*
(?%
inputs?????????
? " ???????????
F__inference_dec_conv1_layer_call_and_return_conditional_losses_9295885l?@7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
+__inference_dec_conv1_layer_call_fn_9295892_?@7?4
-?*
(?%
inputs?????????
? " ???????????
F__inference_dec_conv2_layer_call_and_return_conditional_losses_9295903?LMI?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
+__inference_dec_conv2_layer_call_fn_9295910?LMI?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
F__inference_dec_conv3_layer_call_and_return_conditional_losses_9295952?deI?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
+__inference_dec_conv3_layer_call_fn_9295959?deI?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
C__inference_dense1_layer_call_and_return_conditional_losses_9295830],-0?-
&?#
!?
inputs??????????
? "%?"
?
0????????? 
? |
(__inference_dense1_layer_call_fn_9295837P,-0?-
&?#
!?
inputs??????????
? "?????????? ?
C__inference_dense2_layer_call_and_return_conditional_losses_9295848]34/?,
%?"
 ?
inputs????????? 
? "&?#
?
0??????????
? |
(__inference_dense2_layer_call_fn_9295855P34/?,
%?"
 ?
inputs????????? 
? "????????????
D__inference_dense_2_layer_call_and_return_conditional_losses_9295934\]^/?,
%?"
 ?
inputs?????????#
? "%?"
?
0?????????
? |
)__inference_dense_2_layer_call_fn_9295941O]^/?,
%?"
 ?
inputs?????????#
? "???????????
D__inference_dense_3_layer_call_and_return_conditional_losses_9295969\kl/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? |
)__inference_dense_3_layer_call_fn_9295976Okl/?,
%?"
 ?
inputs?????????
? "???????????
D__inference_flatten_layer_call_and_return_conditional_losses_9295814a7?4
-?*
(?%
inputs?????????
? "&?#
?
0??????????
? ?
)__inference_flatten_layer_call_fn_9295819T7?4
-?*
(?%
inputs?????????
? "????????????
D__inference_model_5_layer_call_and_return_conditional_losses_9295329? !,-34?@LM]^kldel?i
b?_
U?R
+?(
	input_img?????????@@
#? 
	input_act?????????
p

 
? "e?b
[?X
7?4
0/0+???????????????????????????
?
0/1?????????
? ?
D__inference_model_5_layer_call_and_return_conditional_losses_9295367? !,-34?@LM]^kldel?i
b?_
U?R
+?(
	input_img?????????@@
#? 
	input_act?????????
p 

 
? "e?b
[?X
7?4
0/0+???????????????????????????
?
0/1?????????
? ?
D__inference_model_5_layer_call_and_return_conditional_losses_9295622? !,-34?@LM]^kldej?g
`?]
S?P
*?'
inputs/0?????????@@
"?
inputs/1?????????
p

 
? "S?P
I?F
%?"
0/0?????????@@
?
0/1?????????
? ?
D__inference_model_5_layer_call_and_return_conditional_losses_9295720? !,-34?@LM]^kldej?g
`?]
S?P
*?'
inputs/0?????????@@
"?
inputs/1?????????
p 

 
? "S?P
I?F
%?"
0/0?????????@@
?
0/1?????????
? ?
)__inference_model_5_layer_call_fn_9295432? !,-34?@LM]^kldel?i
b?_
U?R
+?(
	input_img?????????@@
#? 
	input_act?????????
p

 
? "W?T
5?2
0+???????????????????????????
?
1??????????
)__inference_model_5_layer_call_fn_9295496? !,-34?@LM]^kldel?i
b?_
U?R
+?(
	input_img?????????@@
#? 
	input_act?????????
p 

 
? "W?T
5?2
0+???????????????????????????
?
1??????????
)__inference_model_5_layer_call_fn_9295746? !,-34?@LM]^kldej?g
`?]
S?P
*?'
inputs/0?????????@@
"?
inputs/1?????????
p

 
? "W?T
5?2
0+???????????????????????????
?
1??????????
)__inference_model_5_layer_call_fn_9295772? !,-34?@LM]^kldej?g
`?]
S?P
*?'
inputs/0?????????@@
"?
inputs/1?????????
p 

 
? "W?T
5?2
0+???????????????????????????
?
1??????????
D__inference_reshape_layer_call_and_return_conditional_losses_9295869a0?-
&?#
!?
inputs??????????
? "-?*
#? 
0?????????
? ?
)__inference_reshape_layer_call_fn_9295874T0?-
&?#
!?
inputs??????????
? " ???????????
%__inference_signature_wrapper_9295524? !,-34?@LM]^kldey?v
? 
o?l
0
	input_act#? 
	input_act?????????
8
	input_img+?(
	input_img?????????@@"k?h
8
	dec_conv3+?(
	dec_conv3?????????@@
,
dense_3!?
dense_3??????????
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_9295034?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
1__inference_up_sampling2d_1_layer_call_fn_9295052?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_9295003?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_up_sampling2d_layer_call_fn_9295021?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84????????????????????????????????????