ƽ
??
:
Add
x"T
y"T
z"T"
Ttype:
2	
A
AddV2
x"T
y"T
z"T"
Ttype:
2	??
E
AssignAddVariableOp
resource
value"dtype"
dtypetype?
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
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
8
DivNoNan
x"T
y"T
z"T"
Ttype:	
2
k
Equal
x"T
y"T
z
""
Ttype:
2	
"$
incompatible_shape_errorbool(?
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
,
Log
x"T
y"T"
Ttype:

2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
:
Maximum
x"T
y"T
z"T"
Ttype:

2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
:
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	?
0
Neg
x"T
y"T"
Ttype:
2
	
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
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
@
ReadVariableOp
resource
value"dtype"
dtypetype?
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
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
0
Sigmoid
x"T
y"T"
Ttype:

2
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
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
;
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
9
VarIsInitializedOp
resource
is_initialized
?"serve*2.3.02unknownϐ
t
embedding_inputPlaceholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
?
5embedding/embeddings/Initializer/random_uniform/shapeConst*'
_class
loc:@embedding/embeddings*
_output_shapes
:*
dtype0*
valueB"??  2   
?
3embedding/embeddings/Initializer/random_uniform/minConst*'
_class
loc:@embedding/embeddings*
_output_shapes
: *
dtype0*
valueB
 *??L?
?
3embedding/embeddings/Initializer/random_uniform/maxConst*'
_class
loc:@embedding/embeddings*
_output_shapes
: *
dtype0*
valueB
 *??L=
?
=embedding/embeddings/Initializer/random_uniform/RandomUniformRandomUniform5embedding/embeddings/Initializer/random_uniform/shape*
T0*'
_class
loc:@embedding/embeddings* 
_output_shapes
:
??2*
dtype0*

seed *
seed2 
?
3embedding/embeddings/Initializer/random_uniform/subSub3embedding/embeddings/Initializer/random_uniform/max3embedding/embeddings/Initializer/random_uniform/min*
T0*'
_class
loc:@embedding/embeddings*
_output_shapes
: 
?
3embedding/embeddings/Initializer/random_uniform/mulMul=embedding/embeddings/Initializer/random_uniform/RandomUniform3embedding/embeddings/Initializer/random_uniform/sub*
T0*'
_class
loc:@embedding/embeddings* 
_output_shapes
:
??2
?
/embedding/embeddings/Initializer/random_uniformAdd3embedding/embeddings/Initializer/random_uniform/mul3embedding/embeddings/Initializer/random_uniform/min*
T0*'
_class
loc:@embedding/embeddings* 
_output_shapes
:
??2
?
embedding/embeddingsVarHandleOp*'
_class
loc:@embedding/embeddings*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:
??2*%
shared_nameembedding/embeddings
y
5embedding/embeddings/IsInitialized/VarIsInitializedOpVarIsInitializedOpembedding/embeddings*
_output_shapes
: 
?
embedding/embeddings/AssignAssignVariableOpembedding/embeddings/embedding/embeddings/Initializer/random_uniform*
dtype0

(embedding/embeddings/Read/ReadVariableOpReadVariableOpembedding/embeddings* 
_output_shapes
:
??2*
dtype0
y
embedding/CastCastembedding_input*

DstT0*

SrcT0*
Truncate( *(
_output_shapes
:??????????
?
embedding/embedding_lookupResourceGatherembedding/embeddingsembedding/Cast*
Tindices0*'
_class
loc:@embedding/embeddings*,
_output_shapes
:??????????2*

batch_dims *
dtype0*
validate_indices(
?
#embedding/embedding_lookup/IdentityIdentityembedding/embedding_lookup*
T0*'
_class
loc:@embedding/embeddings*,
_output_shapes
:??????????2
?
%embedding/embedding_lookup/Identity_1Identity#embedding/embedding_lookup/Identity*
T0*,
_output_shapes
:??????????2
q
/global_average_pooling1d/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
?
global_average_pooling1d/MeanMean%embedding/embedding_lookup/Identity_1/global_average_pooling1d/Mean/reduction_indices*
T0*

Tidx0*'
_output_shapes
:?????????2*
	keep_dims( 
?
-dense/kernel/Initializer/random_uniform/shapeConst*
_class
loc:@dense/kernel*
_output_shapes
:*
dtype0*
valueB"2      
?
+dense/kernel/Initializer/random_uniform/minConst*
_class
loc:@dense/kernel*
_output_shapes
: *
dtype0*
valueB
 *S???
?
+dense/kernel/Initializer/random_uniform/maxConst*
_class
loc:@dense/kernel*
_output_shapes
: *
dtype0*
valueB
 *S??>
?
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
T0*
_class
loc:@dense/kernel*
_output_shapes

:2*
dtype0*

seed *
seed2 
?
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes
: 
?
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@dense/kernel*
_output_shapes

:2
?
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
T0*
_class
loc:@dense/kernel*
_output_shapes

:2
?
dense/kernelVarHandleOp*
_class
loc:@dense/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape
:2*
shared_namedense/kernel
i
-dense/kernel/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense/kernel*
_output_shapes
: 
k
dense/kernel/AssignAssignVariableOpdense/kernel'dense/kernel/Initializer/random_uniform*
dtype0
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:2*
dtype0
?
dense/bias/Initializer/zerosConst*
_class
loc:@dense/bias*
_output_shapes
:*
dtype0*
valueB*    
?

dense/biasVarHandleOp*
_class
loc:@dense/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*
shared_name
dense/bias
e
+dense/bias/IsInitialized/VarIsInitializedOpVarIsInitializedOp
dense/bias*
_output_shapes
: 
\
dense/bias/AssignAssignVariableOp
dense/biasdense/bias/Initializer/zeros*
dtype0
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
h
dense/MatMul/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:2*
dtype0
?
dense/MatMulMatMulglobal_average_pooling1d/Meandense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????*
transpose_a( *
transpose_b( 
c
dense/BiasAdd/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
?
dense/BiasAddBiasAdddense/MatMuldense/BiasAdd/ReadVariableOp*
T0*'
_output_shapes
:?????????*
data_formatNHWC
Y
dense/SigmoidSigmoiddense/BiasAdd*
T0*'
_output_shapes
:?????????
?
PlaceholderPlaceholder*0
_output_shapes
:??????????????????*
dtype0*%
shape:??????????????????
T
AssignVariableOpAssignVariableOpembedding/embeddingsPlaceholder*
dtype0
x
ReadVariableOpReadVariableOpembedding/embeddings^AssignVariableOp* 
_output_shapes
:
??2*
dtype0
?
Placeholder_1Placeholder*0
_output_shapes
:??????????????????*
dtype0*%
shape:??????????????????
P
AssignVariableOp_1AssignVariableOpdense/kernelPlaceholder_1*
dtype0
r
ReadVariableOp_1ReadVariableOpdense/kernel^AssignVariableOp_1*
_output_shapes

:2*
dtype0
h
Placeholder_2Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
N
AssignVariableOp_2AssignVariableOp
dense/biasPlaceholder_2*
dtype0
l
ReadVariableOp_2ReadVariableOp
dense/bias^AssignVariableOp_2*
_output_shapes
:*
dtype0
V
VarIsInitializedOpVarIsInitializedOpembedding/embeddings*
_output_shapes
: 
N
VarIsInitializedOp_1VarIsInitializedOp
dense/bias*
_output_shapes
: 
P
VarIsInitializedOp_2VarIsInitializedOpdense/kernel*
_output_shapes
: 
T
initNoOp^dense/bias/Assign^dense/kernel/Assign^embedding/embeddings/Assign
?
dense_targetPlaceholder*0
_output_shapes
:??????????????????*
dtype0*%
shape:??????????????????
v
total/Initializer/zerosConst*
_class

loc:@total*
_output_shapes
: *
dtype0*
valueB
 *    
?
totalVarHandleOp*
_class

loc:@total*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape: *
shared_nametotal
[
&total/IsInitialized/VarIsInitializedOpVarIsInitializedOptotal*
_output_shapes
: 
M
total/AssignAssignVariableOptotaltotal/Initializer/zeros*
dtype0
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
v
count/Initializer/zerosConst*
_class

loc:@count*
_output_shapes
: *
dtype0*
valueB
 *    
?
countVarHandleOp*
_class

loc:@count*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape: *
shared_namecount
[
&count/IsInitialized/VarIsInitializedOpVarIsInitializedOpcount*
_output_shapes
: 
M
count/AssignAssignVariableOpcountcount/Initializer/zeros*
dtype0
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
\
metrics/accuracy/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?
}
metrics/accuracy/GreaterGreaterdense/Sigmoidmetrics/accuracy/Cast/x*
T0*'
_output_shapes
:?????????
?
metrics/accuracy/Cast_1Castmetrics/accuracy/Greater*

DstT0*

SrcT0
*
Truncate( *'
_output_shapes
:?????????
?
metrics/accuracy/EqualEqualdense_targetmetrics/accuracy/Cast_1*
T0*0
_output_shapes
:??????????????????*
incompatible_shape_error(
?
metrics/accuracy/Cast_2Castmetrics/accuracy/Equal*

DstT0*

SrcT0
*
Truncate( *0
_output_shapes
:??????????????????
r
'metrics/accuracy/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????
?
metrics/accuracy/MeanMeanmetrics/accuracy/Cast_2'metrics/accuracy/Mean/reduction_indices*
T0*

Tidx0*#
_output_shapes
:?????????*
	keep_dims( 
`
metrics/accuracy/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
?
metrics/accuracy/SumSummetrics/accuracy/Meanmetrics/accuracy/Const*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
e
$metrics/accuracy/AssignAddVariableOpAssignAddVariableOptotalmetrics/accuracy/Sum*
dtype0
?
metrics/accuracy/ReadVariableOpReadVariableOptotal%^metrics/accuracy/AssignAddVariableOp^metrics/accuracy/Sum*
_output_shapes
: *
dtype0
e
metrics/accuracy/SizeSizemetrics/accuracy/Mean*
T0*
_output_shapes
: *
out_type0
v
metrics/accuracy/Cast_3Castmetrics/accuracy/Size*

DstT0*

SrcT0*
Truncate( *
_output_shapes
: 
?
&metrics/accuracy/AssignAddVariableOp_1AssignAddVariableOpcountmetrics/accuracy/Cast_3%^metrics/accuracy/AssignAddVariableOp*
dtype0
?
!metrics/accuracy/ReadVariableOp_1ReadVariableOpcount%^metrics/accuracy/AssignAddVariableOp'^metrics/accuracy/AssignAddVariableOp_1*
_output_shapes
: *
dtype0
?
*metrics/accuracy/div_no_nan/ReadVariableOpReadVariableOptotal'^metrics/accuracy/AssignAddVariableOp_1*
_output_shapes
: *
dtype0
?
,metrics/accuracy/div_no_nan/ReadVariableOp_1ReadVariableOpcount'^metrics/accuracy/AssignAddVariableOp_1*
_output_shapes
: *
dtype0
?
metrics/accuracy/div_no_nanDivNoNan*metrics/accuracy/div_no_nan/ReadVariableOp,metrics/accuracy/div_no_nan/ReadVariableOp_1*
T0*
_output_shapes
: 
c
metrics/accuracy/IdentityIdentitymetrics/accuracy/div_no_nan*
T0*
_output_shapes
: 
Z
loss/dense_loss/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
\
loss/dense_loss/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *???3
Z
loss/dense_loss/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
k
loss/dense_loss/subSubloss/dense_loss/sub/xloss/dense_loss/Const_1*
T0*
_output_shapes
: 
?
%loss/dense_loss/clip_by_value/MinimumMinimumdense/Sigmoidloss/dense_loss/sub*
T0*'
_output_shapes
:?????????
?
loss/dense_loss/clip_by_valueMaximum%loss/dense_loss/clip_by_value/Minimumloss/dense_loss/Const_1*
T0*'
_output_shapes
:?????????
Z
loss/dense_loss/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3
?
loss/dense_loss/addAddV2loss/dense_loss/clip_by_valueloss/dense_loss/add/y*
T0*'
_output_shapes
:?????????
a
loss/dense_loss/LogLogloss/dense_loss/add*
T0*'
_output_shapes
:?????????
x
loss/dense_loss/mulMuldense_targetloss/dense_loss/Log*
T0*0
_output_shapes
:??????????????????
\
loss/dense_loss/sub_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
~
loss/dense_loss/sub_1Subloss/dense_loss/sub_1/xdense_target*
T0*0
_output_shapes
:??????????????????
\
loss/dense_loss/sub_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
loss/dense_loss/sub_2Subloss/dense_loss/sub_2/xloss/dense_loss/clip_by_value*
T0*'
_output_shapes
:?????????
\
loss/dense_loss/add_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *???3
?
loss/dense_loss/add_1AddV2loss/dense_loss/sub_2loss/dense_loss/add_1/y*
T0*'
_output_shapes
:?????????
e
loss/dense_loss/Log_1Logloss/dense_loss/add_1*
T0*'
_output_shapes
:?????????
?
loss/dense_loss/mul_1Mulloss/dense_loss/sub_1loss/dense_loss/Log_1*
T0*0
_output_shapes
:??????????????????
?
loss/dense_loss/add_2AddV2loss/dense_loss/mulloss/dense_loss/mul_1*
T0*0
_output_shapes
:??????????????????
l
loss/dense_loss/NegNegloss/dense_loss/add_2*
T0*0
_output_shapes
:??????????????????
q
&loss/dense_loss/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
valueB :
?????????
?
loss/dense_loss/MeanMeanloss/dense_loss/Neg&loss/dense_loss/Mean/reduction_indices*
T0*

Tidx0*#
_output_shapes
:?????????*
	keep_dims( 
h
#loss/dense_loss/weighted_loss/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
!loss/dense_loss/weighted_loss/MulMulloss/dense_loss/Mean#loss/dense_loss/weighted_loss/Const*
T0*#
_output_shapes
:?????????
a
loss/dense_loss/Const_2Const*
_output_shapes
:*
dtype0*
valueB: 
?
loss/dense_loss/SumSum!loss/dense_loss/weighted_loss/Mulloss/dense_loss/Const_2*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
x
loss/dense_loss/num_elementsSize!loss/dense_loss/weighted_loss/Mul*
T0*
_output_shapes
: *
out_type0
?
!loss/dense_loss/num_elements/CastCastloss/dense_loss/num_elements*

DstT0*

SrcT0*
Truncate( *
_output_shapes
: 
Z
loss/dense_loss/Const_3Const*
_output_shapes
: *
dtype0*
valueB 
?
loss/dense_loss/Sum_1Sumloss/dense_loss/Sumloss/dense_loss/Const_3*
T0*

Tidx0*
_output_shapes
: *
	keep_dims( 
|
loss/dense_loss/valueDivNoNanloss/dense_loss/Sum_1!loss/dense_loss/num_elements/Cast*
T0*
_output_shapes
: 
O

loss/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
S
loss/mulMul
loss/mul/xloss/dense_loss/value*
T0*
_output_shapes
: 
q
iter/Initializer/zerosConst*
_class
	loc:@iter*
_output_shapes
: *
dtype0	*
value	B	 R 
?
iterVarHandleOp*
_class
	loc:@iter*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0	*
shape: *
shared_nameiter
Y
%iter/IsInitialized/VarIsInitializedOpVarIsInitializedOpiter*
_output_shapes
: 
J
iter/AssignAssignVariableOpiteriter/Initializer/zeros*
dtype0	
U
iter/Read/ReadVariableOpReadVariableOpiter*
_output_shapes
: *
dtype0	
?
 beta_1/Initializer/initial_valueConst*
_class
loc:@beta_1*
_output_shapes
: *
dtype0*
valueB
 *fff?
?
beta_1VarHandleOp*
_class
loc:@beta_1*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape: *
shared_namebeta_1
]
'beta_1/IsInitialized/VarIsInitializedOpVarIsInitializedOpbeta_1*
_output_shapes
: 
X
beta_1/AssignAssignVariableOpbeta_1 beta_1/Initializer/initial_value*
dtype0
Y
beta_1/Read/ReadVariableOpReadVariableOpbeta_1*
_output_shapes
: *
dtype0
?
 beta_2/Initializer/initial_valueConst*
_class
loc:@beta_2*
_output_shapes
: *
dtype0*
valueB
 *w??
?
beta_2VarHandleOp*
_class
loc:@beta_2*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape: *
shared_namebeta_2
]
'beta_2/IsInitialized/VarIsInitializedOpVarIsInitializedOpbeta_2*
_output_shapes
: 
X
beta_2/AssignAssignVariableOpbeta_2 beta_2/Initializer/initial_value*
dtype0
Y
beta_2/Read/ReadVariableOpReadVariableOpbeta_2*
_output_shapes
: *
dtype0
~
decay/Initializer/initial_valueConst*
_class

loc:@decay*
_output_shapes
: *
dtype0*
valueB
 *    
?
decayVarHandleOp*
_class

loc:@decay*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape: *
shared_namedecay
[
&decay/IsInitialized/VarIsInitializedOpVarIsInitializedOpdecay*
_output_shapes
: 
U
decay/AssignAssignVariableOpdecaydecay/Initializer/initial_value*
dtype0
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
?
'learning_rate/Initializer/initial_valueConst* 
_class
loc:@learning_rate*
_output_shapes
: *
dtype0*
valueB
 *o?:
?
learning_rateVarHandleOp* 
_class
loc:@learning_rate*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape: *
shared_namelearning_rate
k
.learning_rate/IsInitialized/VarIsInitializedOpVarIsInitializedOplearning_rate*
_output_shapes
: 
m
learning_rate/AssignAssignVariableOplearning_rate'learning_rate/Initializer/initial_value*
dtype0
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
?
8embedding/embeddings/m/Initializer/zeros/shape_as_tensorConst*'
_class
loc:@embedding/embeddings*
_output_shapes
:*
dtype0*
valueB"??  2   
?
.embedding/embeddings/m/Initializer/zeros/ConstConst*'
_class
loc:@embedding/embeddings*
_output_shapes
: *
dtype0*
valueB
 *    
?
(embedding/embeddings/m/Initializer/zerosFill8embedding/embeddings/m/Initializer/zeros/shape_as_tensor.embedding/embeddings/m/Initializer/zeros/Const*
T0*'
_class
loc:@embedding/embeddings* 
_output_shapes
:
??2*

index_type0
?
embedding/embeddings/mVarHandleOp*'
_class
loc:@embedding/embeddings*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:
??2*'
shared_nameembedding/embeddings/m
?
7embedding/embeddings/m/IsInitialized/VarIsInitializedOpVarIsInitializedOpembedding/embeddings/m*'
_class
loc:@embedding/embeddings*
_output_shapes
: 
?
embedding/embeddings/m/AssignAssignVariableOpembedding/embeddings/m(embedding/embeddings/m/Initializer/zeros*
dtype0
?
*embedding/embeddings/m/Read/ReadVariableOpReadVariableOpembedding/embeddings/m*'
_class
loc:@embedding/embeddings* 
_output_shapes
:
??2*
dtype0
?
 dense/kernel/m/Initializer/zerosConst*
_class
loc:@dense/kernel*
_output_shapes

:2*
dtype0*
valueB2*    
?
dense/kernel/mVarHandleOp*
_class
loc:@dense/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape
:2*
shared_namedense/kernel/m
?
/dense/kernel/m/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense/kernel/m*
_class
loc:@dense/kernel*
_output_shapes
: 
h
dense/kernel/m/AssignAssignVariableOpdense/kernel/m dense/kernel/m/Initializer/zeros*
dtype0
?
"dense/kernel/m/Read/ReadVariableOpReadVariableOpdense/kernel/m*
_class
loc:@dense/kernel*
_output_shapes

:2*
dtype0
?
dense/bias/m/Initializer/zerosConst*
_class
loc:@dense/bias*
_output_shapes
:*
dtype0*
valueB*    
?
dense/bias/mVarHandleOp*
_class
loc:@dense/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*
shared_namedense/bias/m
?
-dense/bias/m/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense/bias/m*
_class
loc:@dense/bias*
_output_shapes
: 
b
dense/bias/m/AssignAssignVariableOpdense/bias/mdense/bias/m/Initializer/zeros*
dtype0
?
 dense/bias/m/Read/ReadVariableOpReadVariableOpdense/bias/m*
_class
loc:@dense/bias*
_output_shapes
:*
dtype0
?
8embedding/embeddings/v/Initializer/zeros/shape_as_tensorConst*'
_class
loc:@embedding/embeddings*
_output_shapes
:*
dtype0*
valueB"??  2   
?
.embedding/embeddings/v/Initializer/zeros/ConstConst*'
_class
loc:@embedding/embeddings*
_output_shapes
: *
dtype0*
valueB
 *    
?
(embedding/embeddings/v/Initializer/zerosFill8embedding/embeddings/v/Initializer/zeros/shape_as_tensor.embedding/embeddings/v/Initializer/zeros/Const*
T0*'
_class
loc:@embedding/embeddings* 
_output_shapes
:
??2*

index_type0
?
embedding/embeddings/vVarHandleOp*'
_class
loc:@embedding/embeddings*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:
??2*'
shared_nameembedding/embeddings/v
?
7embedding/embeddings/v/IsInitialized/VarIsInitializedOpVarIsInitializedOpembedding/embeddings/v*'
_class
loc:@embedding/embeddings*
_output_shapes
: 
?
embedding/embeddings/v/AssignAssignVariableOpembedding/embeddings/v(embedding/embeddings/v/Initializer/zeros*
dtype0
?
*embedding/embeddings/v/Read/ReadVariableOpReadVariableOpembedding/embeddings/v*'
_class
loc:@embedding/embeddings* 
_output_shapes
:
??2*
dtype0
?
 dense/kernel/v/Initializer/zerosConst*
_class
loc:@dense/kernel*
_output_shapes

:2*
dtype0*
valueB2*    
?
dense/kernel/vVarHandleOp*
_class
loc:@dense/kernel*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape
:2*
shared_namedense/kernel/v
?
/dense/kernel/v/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense/kernel/v*
_class
loc:@dense/kernel*
_output_shapes
: 
h
dense/kernel/v/AssignAssignVariableOpdense/kernel/v dense/kernel/v/Initializer/zeros*
dtype0
?
"dense/kernel/v/Read/ReadVariableOpReadVariableOpdense/kernel/v*
_class
loc:@dense/kernel*
_output_shapes

:2*
dtype0
?
dense/bias/v/Initializer/zerosConst*
_class
loc:@dense/bias*
_output_shapes
:*
dtype0*
valueB*    
?
dense/bias/vVarHandleOp*
_class
loc:@dense/bias*
_output_shapes
: *
allowed_devices
 *
	container *
dtype0*
shape:*
shared_namedense/bias/v
?
-dense/bias/v/IsInitialized/VarIsInitializedOpVarIsInitializedOpdense/bias/v*
_class
loc:@dense/bias*
_output_shapes
: 
b
dense/bias/v/AssignAssignVariableOpdense/bias/vdense/bias/v/Initializer/zeros*
dtype0
?
 dense/bias/v/Read/ReadVariableOpReadVariableOpdense/bias/v*
_class
loc:@dense/bias*
_output_shapes
:*
dtype0
I
VarIsInitializedOp_3VarIsInitializedOptotal*
_output_shapes
: 
I
VarIsInitializedOp_4VarIsInitializedOpcount*
_output_shapes
: 
H
VarIsInitializedOp_5VarIsInitializedOpiter*
_output_shapes
: 
J
VarIsInitializedOp_6VarIsInitializedOpbeta_1*
_output_shapes
: 
I
VarIsInitializedOp_7VarIsInitializedOpdecay*
_output_shapes
: 
Z
VarIsInitializedOp_8VarIsInitializedOpembedding/embeddings/m*
_output_shapes
: 
P
VarIsInitializedOp_9VarIsInitializedOpdense/bias/v*
_output_shapes
: 
K
VarIsInitializedOp_10VarIsInitializedOpbeta_2*
_output_shapes
: 
R
VarIsInitializedOp_11VarIsInitializedOplearning_rate*
_output_shapes
: 
S
VarIsInitializedOp_12VarIsInitializedOpdense/kernel/m*
_output_shapes
: 
Q
VarIsInitializedOp_13VarIsInitializedOpdense/bias/m*
_output_shapes
: 
[
VarIsInitializedOp_14VarIsInitializedOpembedding/embeddings/v*
_output_shapes
: 
S
VarIsInitializedOp_15VarIsInitializedOpdense/kernel/v*
_output_shapes
: 
?
init_1NoOp^beta_1/Assign^beta_2/Assign^count/Assign^decay/Assign^dense/bias/m/Assign^dense/bias/v/Assign^dense/kernel/m/Assign^dense/kernel/v/Assign^embedding/embeddings/m/Assign^embedding/embeddings/v/Assign^iter/Assign^learning_rate/Assign^total/Assign
N
Placeholder_3Placeholder*
_output_shapes
: *
dtype0	*
shape: 
H
AssignVariableOp_3AssignVariableOpiterPlaceholder_3*
dtype0	
b
ReadVariableOp_3ReadVariableOpiter^AssignVariableOp_3*
_output_shapes
: *
dtype0	
?
Placeholder_4Placeholder*0
_output_shapes
:??????????????????*
dtype0*%
shape:??????????????????
Z
AssignVariableOp_4AssignVariableOpembedding/embeddings/mPlaceholder_4*
dtype0
~
ReadVariableOp_4ReadVariableOpembedding/embeddings/m^AssignVariableOp_4* 
_output_shapes
:
??2*
dtype0
?
Placeholder_5Placeholder*0
_output_shapes
:??????????????????*
dtype0*%
shape:??????????????????
R
AssignVariableOp_5AssignVariableOpdense/kernel/mPlaceholder_5*
dtype0
t
ReadVariableOp_5ReadVariableOpdense/kernel/m^AssignVariableOp_5*
_output_shapes

:2*
dtype0
h
Placeholder_6Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
P
AssignVariableOp_6AssignVariableOpdense/bias/mPlaceholder_6*
dtype0
n
ReadVariableOp_6ReadVariableOpdense/bias/m^AssignVariableOp_6*
_output_shapes
:*
dtype0
?
Placeholder_7Placeholder*0
_output_shapes
:??????????????????*
dtype0*%
shape:??????????????????
Z
AssignVariableOp_7AssignVariableOpembedding/embeddings/vPlaceholder_7*
dtype0
~
ReadVariableOp_7ReadVariableOpembedding/embeddings/v^AssignVariableOp_7* 
_output_shapes
:
??2*
dtype0
?
Placeholder_8Placeholder*0
_output_shapes
:??????????????????*
dtype0*%
shape:??????????????????
R
AssignVariableOp_8AssignVariableOpdense/kernel/vPlaceholder_8*
dtype0
t
ReadVariableOp_8ReadVariableOpdense/kernel/v^AssignVariableOp_8*
_output_shapes

:2*
dtype0
h
Placeholder_9Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
P
AssignVariableOp_9AssignVariableOpdense/bias/vPlaceholder_9*
dtype0
n
ReadVariableOp_9ReadVariableOpdense/bias/v^AssignVariableOp_9*
_output_shapes
:*
dtype0
]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
_
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
strided_sliceStridedSliceembedding_inputstrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
Index0*
T0*
_output_shapes	
:?*

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
?
strided_slice_1StridedSliceembedding_inputstrided_slice_1/stackstrided_slice_1/stack_1strided_slice_1/stack_2*
Index0*
T0*
_output_shapes	
:?*

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
Y
save/filename/inputConst*
_output_shapes
: *
dtype0*
valueB Bmodel
n
save/filenamePlaceholderWithDefaultsave/filename/input*
_output_shapes
: *
dtype0*
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
_output_shapes
: *
dtype0*
shape: 
{
save/StaticRegexFullMatchStaticRegexFullMatch
save/Const"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*
a
save/Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part
?
save/Const_2Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_b2d9295e28ab40ceb908fd962461b10f/part
|
save/SelectSelectsave/StaticRegexFullMatchsave/Const_1save/Const_2"/device:CPU:**
T0*
_output_shapes
: 
w
save/StringJoin
StringJoin
save/Constsave/Select"/device:CPU:**
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
k
save/ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
?
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
?
save/SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?Bbeta_1Bbeta_2BdecayB
dense/biasBdense/bias/mBdense/bias/vBdense/kernelBdense/kernel/mBdense/kernel/vBembedding/embeddingsBembedding/embeddings/mBembedding/embeddings/vBiterBlearning_rate
?
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B 
?
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOpdense/bias/Read/ReadVariableOp dense/bias/m/Read/ReadVariableOp dense/bias/v/Read/ReadVariableOp dense/kernel/Read/ReadVariableOp"dense/kernel/m/Read/ReadVariableOp"dense/kernel/v/Read/ReadVariableOp(embedding/embeddings/Read/ReadVariableOp*embedding/embeddings/m/Read/ReadVariableOp*embedding/embeddings/v/Read/ReadVariableOpiter/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp"/device:CPU:0*
dtypes
2	
?
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
?
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
N*
T0*
_output_shapes
:*

axis 
?
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(
?
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
?
save/RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?Bbeta_1Bbeta_2BdecayB
dense/biasBdense/bias/mBdense/bias/vBdense/kernelBdense/kernel/mBdense/kernel/vBembedding/embeddingsBembedding/embeddings/mBembedding/embeddings/vBiterBlearning_rate
?
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*/
value&B$B B B B B B B B B B B B B B 
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*L
_output_shapes:
8::::::::::::::*
dtypes
2	
N
save/Identity_1Identitysave/RestoreV2*
T0*
_output_shapes
:
O
save/AssignVariableOpAssignVariableOpbeta_1save/Identity_1*
dtype0
P
save/Identity_2Identitysave/RestoreV2:1*
T0*
_output_shapes
:
Q
save/AssignVariableOp_1AssignVariableOpbeta_2save/Identity_2*
dtype0
P
save/Identity_3Identitysave/RestoreV2:2*
T0*
_output_shapes
:
P
save/AssignVariableOp_2AssignVariableOpdecaysave/Identity_3*
dtype0
P
save/Identity_4Identitysave/RestoreV2:3*
T0*
_output_shapes
:
U
save/AssignVariableOp_3AssignVariableOp
dense/biassave/Identity_4*
dtype0
P
save/Identity_5Identitysave/RestoreV2:4*
T0*
_output_shapes
:
W
save/AssignVariableOp_4AssignVariableOpdense/bias/msave/Identity_5*
dtype0
P
save/Identity_6Identitysave/RestoreV2:5*
T0*
_output_shapes
:
W
save/AssignVariableOp_5AssignVariableOpdense/bias/vsave/Identity_6*
dtype0
P
save/Identity_7Identitysave/RestoreV2:6*
T0*
_output_shapes
:
W
save/AssignVariableOp_6AssignVariableOpdense/kernelsave/Identity_7*
dtype0
P
save/Identity_8Identitysave/RestoreV2:7*
T0*
_output_shapes
:
Y
save/AssignVariableOp_7AssignVariableOpdense/kernel/msave/Identity_8*
dtype0
P
save/Identity_9Identitysave/RestoreV2:8*
T0*
_output_shapes
:
Y
save/AssignVariableOp_8AssignVariableOpdense/kernel/vsave/Identity_9*
dtype0
Q
save/Identity_10Identitysave/RestoreV2:9*
T0*
_output_shapes
:
`
save/AssignVariableOp_9AssignVariableOpembedding/embeddingssave/Identity_10*
dtype0
R
save/Identity_11Identitysave/RestoreV2:10*
T0*
_output_shapes
:
c
save/AssignVariableOp_10AssignVariableOpembedding/embeddings/msave/Identity_11*
dtype0
R
save/Identity_12Identitysave/RestoreV2:11*
T0*
_output_shapes
:
c
save/AssignVariableOp_11AssignVariableOpembedding/embeddings/vsave/Identity_12*
dtype0
R
save/Identity_13Identitysave/RestoreV2:12*
T0	*
_output_shapes
:
Q
save/AssignVariableOp_12AssignVariableOpitersave/Identity_13*
dtype0	
R
save/Identity_14Identitysave/RestoreV2:13*
T0*
_output_shapes
:
Z
save/AssignVariableOp_13AssignVariableOplearning_ratesave/Identity_14*
dtype0
?
save/restore_shardNoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_10^save/AssignVariableOp_11^save/AssignVariableOp_12^save/AssignVariableOp_13^save/AssignVariableOp_2^save/AssignVariableOp_3^save/AssignVariableOp_4^save/AssignVariableOp_5^save/AssignVariableOp_6^save/AssignVariableOp_7^save/AssignVariableOp_8^save/AssignVariableOp_9
-
save/restore_allNoOp^save/restore_shard"?<
save/Const:0save/Identity:0save/restore_all (5 @F8"?
trainable_variables??
?
embedding/embeddings:0embedding/embeddings/Assign*embedding/embeddings/Read/ReadVariableOp:0(21embedding/embeddings/Initializer/random_uniform:08
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08"?
	variables??
?
embedding/embeddings:0embedding/embeddings/Assign*embedding/embeddings/Read/ReadVariableOp:0(21embedding/embeddings/Initializer/random_uniform:08
x
dense/kernel:0dense/kernel/Assign"dense/kernel/Read/ReadVariableOp:0(2)dense/kernel/Initializer/random_uniform:08
g
dense/bias:0dense/bias/Assign dense/bias/Read/ReadVariableOp:0(2dense/bias/Initializer/zeros:08
O
iter:0iter/Assigniter/Read/ReadVariableOp:0(2iter/Initializer/zeros:0H
_
beta_1:0beta_1/Assignbeta_1/Read/ReadVariableOp:0(2"beta_1/Initializer/initial_value:0H
_
beta_2:0beta_2/Assignbeta_2/Read/ReadVariableOp:0(2"beta_2/Initializer/initial_value:0H
[
decay:0decay/Assigndecay/Read/ReadVariableOp:0(2!decay/Initializer/initial_value:0H
{
learning_rate:0learning_rate/Assign#learning_rate/Read/ReadVariableOp:0(2)learning_rate/Initializer/initial_value:0H
?
embedding/embeddings/m:0embedding/embeddings/m/Assign,embedding/embeddings/m/Read/ReadVariableOp:0(2*embedding/embeddings/m/Initializer/zeros:0
u
dense/kernel/m:0dense/kernel/m/Assign$dense/kernel/m/Read/ReadVariableOp:0(2"dense/kernel/m/Initializer/zeros:0
m
dense/bias/m:0dense/bias/m/Assign"dense/bias/m/Read/ReadVariableOp:0(2 dense/bias/m/Initializer/zeros:0
?
embedding/embeddings/v:0embedding/embeddings/v/Assign,embedding/embeddings/v/Read/ReadVariableOp:0(2*embedding/embeddings/v/Initializer/zeros:0
u
dense/kernel/v:0dense/kernel/v/Assign$dense/kernel/v/Read/ReadVariableOp:0(2"dense/kernel/v/Initializer/zeros:0
m
dense/bias/v:0dense/bias/v/Assign"dense/bias/v/Read/ReadVariableOp:0(2 dense/bias/v/Initializer/zeros:0*|
serving_defaulti
#
input
strided_slice:0?&
income
strided_slice_1:0?tensorflow/serving/predict