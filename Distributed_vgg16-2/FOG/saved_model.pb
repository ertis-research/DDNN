Ż˘
ĚŁ
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
ž
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
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.12v2.3.1-0-gfcc4b966f18ç
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
@*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
@*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:*
dtype0
z
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_2/kernel
s
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel* 
_output_shapes
:
*
dtype0
q
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes	
:*
dtype0

fog_output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*"
shared_namefog_output/kernel
x
%fog_output/kernel/Read/ReadVariableOpReadVariableOpfog_output/kernel*
_output_shapes
:	
*
dtype0
v
fog_output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namefog_output/bias
o
#fog_output/bias/Read/ReadVariableOpReadVariableOpfog_output/bias*
_output_shapes
:
*
dtype0

block2_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameblock2_conv1/kernel

'block2_conv1/kernel/Read/ReadVariableOpReadVariableOpblock2_conv1/kernel*'
_output_shapes
:@*
dtype0
{
block2_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock2_conv1/bias
t
%block2_conv1/bias/Read/ReadVariableOpReadVariableOpblock2_conv1/bias*
_output_shapes	
:*
dtype0

block2_conv2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameblock2_conv2/kernel

'block2_conv2/kernel/Read/ReadVariableOpReadVariableOpblock2_conv2/kernel*(
_output_shapes
:*
dtype0
{
block2_conv2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameblock2_conv2/bias
t
%block2_conv2/bias/Read/ReadVariableOpReadVariableOpblock2_conv2/bias*
_output_shapes	
:*
dtype0

NoOpNoOp
Š!
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ä 
valueÚ B×  BĐ 

layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
regularization_losses
trainable_variables
		variables

	keras_api

signatures
 
ş
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
 regularization_losses
!trainable_variables
"	variables
#	keras_api
h

$kernel
%bias
&regularization_losses
'trainable_variables
(	variables
)	keras_api
 
*
0
1
2
3
$4
%5
F
*0
+1
,2
-3
4
5
6
7
$8
%9
­
regularization_losses

.layers
/non_trainable_variables
0layer_regularization_losses
1metrics
2layer_metrics
trainable_variables
		variables
 
 
h

*kernel
+bias
3regularization_losses
4trainable_variables
5	variables
6	keras_api
h

,kernel
-bias
7regularization_losses
8trainable_variables
9	variables
:	keras_api
R
;regularization_losses
<trainable_variables
=	variables
>	keras_api
 
 

*0
+1
,2
-3
­
regularization_losses

?layers
@non_trainable_variables
Alayer_regularization_losses
Bmetrics
Clayer_metrics
trainable_variables
	variables
 
 
 
­
regularization_losses

Dlayers
Enon_trainable_variables
Flayer_regularization_losses
Gmetrics
Hlayer_metrics
trainable_variables
	variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
regularization_losses

Ilayers
Jnon_trainable_variables
Klayer_regularization_losses
Lmetrics
Mlayer_metrics
trainable_variables
	variables
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
 regularization_losses

Nlayers
Onon_trainable_variables
Player_regularization_losses
Qmetrics
Rlayer_metrics
!trainable_variables
"	variables
][
VARIABLE_VALUEfog_output/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEfog_output/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

$0
%1

$0
%1
­
&regularization_losses

Slayers
Tnon_trainable_variables
Ulayer_regularization_losses
Vmetrics
Wlayer_metrics
'trainable_variables
(	variables
OM
VARIABLE_VALUEblock2_conv1/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEblock2_conv1/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEblock2_conv2/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEblock2_conv2/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
*
0
1
2
3
4
5

*0
+1
,2
-3
 
 
 
 
 

*0
+1
­
3regularization_losses

Xlayers
Ynon_trainable_variables
Zlayer_regularization_losses
[metrics
\layer_metrics
4trainable_variables
5	variables
 
 

,0
-1
­
7regularization_losses

]layers
^non_trainable_variables
_layer_regularization_losses
`metrics
alayer_metrics
8trainable_variables
9	variables
 
 
 
­
;regularization_losses

blayers
cnon_trainable_variables
dlayer_regularization_losses
emetrics
flayer_metrics
<trainable_variables
=	variables

0
1
2
3

*0
+1
,2
-3
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

*0
+1
 
 
 
 

,0
-1
 
 
 
 
 
 
 
 

serving_default_input_5Placeholder*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
dtype0*$
shape:˙˙˙˙˙˙˙˙˙@

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_5block2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasfog_output/kernelfog_output/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *C
_output_shapes1
/:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8 *-
f(R&
$__inference_signature_wrapper_437286
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
˘
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp%fog_output/kernel/Read/ReadVariableOp#fog_output/bias/Read/ReadVariableOp'block2_conv1/kernel/Read/ReadVariableOp%block2_conv1/bias/Read/ReadVariableOp'block2_conv2/kernel/Read/ReadVariableOp%block2_conv2/bias/Read/ReadVariableOpConst*
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
GPU2 *0J 8 *(
f#R!
__inference__traced_save_437655
Ő
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_1/kerneldense_1/biasdense_2/kerneldense_2/biasfog_output/kernelfog_output/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/bias*
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
GPU2 *0J 8 *+
f&R$
"__inference__traced_restore_437695ĂŁ
ż
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_437496

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙@2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙@2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
­
F
*__inference_flatten_1_layer_call_fn_437501

inputs
identityÉ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_4370212
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙@2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ą-
­
"__inference__traced_restore_437695
file_prefix#
assignvariableop_dense_1_kernel#
assignvariableop_1_dense_1_bias%
!assignvariableop_2_dense_2_kernel#
assignvariableop_3_dense_2_bias(
$assignvariableop_4_fog_output_kernel&
"assignvariableop_5_fog_output_bias*
&assignvariableop_6_block2_conv1_kernel(
$assignvariableop_7_block2_conv1_bias*
&assignvariableop_8_block2_conv2_kernel(
$assignvariableop_9_block2_conv2_bias
identity_11˘AssignVariableOp˘AssignVariableOp_1˘AssignVariableOp_2˘AssignVariableOp_3˘AssignVariableOp_4˘AssignVariableOp_5˘AssignVariableOp_6˘AssignVariableOp_7˘AssignVariableOp_8˘AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names¤
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
RestoreV2/shape_and_slicesâ
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

Identity
AssignVariableOpAssignVariableOpassignvariableop_dense_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¤
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ś
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¤
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4Š
AssignVariableOp_4AssignVariableOp$assignvariableop_4_fog_output_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5§
AssignVariableOp_5AssignVariableOp"assignvariableop_5_fog_output_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6Ť
AssignVariableOp_6AssignVariableOp&assignvariableop_6_block2_conv1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Š
AssignVariableOp_7AssignVariableOp$assignvariableop_7_block2_conv1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Ť
AssignVariableOp_8AssignVariableOp&assignvariableop_8_block2_conv2_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Š
AssignVariableOp_9AssignVariableOp$assignvariableop_9_block2_conv2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpş
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_10­
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
Ş
H
,__inference_block2_pool_layer_call_fn_436844

inputs
identityí
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_block2_pool_layer_call_and_return_conditional_losses_4368382
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:r n
J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ů


$__inference_fog_layer_call_fn_437200
input_5
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity

identity_1˘StatefulPartitionedCallú
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *C
_output_shapes1
/:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8 *H
fCRA
?__inference_fog_layer_call_and_return_conditional_losses_4371752
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*V
_input_shapesE
C:˙˙˙˙˙˙˙˙˙@::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
!
_user_specified_name	input_5
˝

$__inference_fog_layer_call_fn_436976
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *H
fCRA
?__inference_fog_layer_call_and_return_conditional_losses_4369652
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙@::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
!
_user_specified_name	input_2
č

+__inference_fog_output_layer_call_fn_437561

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallű
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_fog_output_layer_call_and_return_conditional_losses_4370942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
	
°
H__inference_block2_conv2_layer_call_and_return_conditional_losses_436886

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:˙˙˙˙˙˙˙˙˙:::X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ę

?__inference_fog_layer_call_and_return_conditional_losses_436904
input_2
block2_conv1_436870
block2_conv1_436872
block2_conv2_436897
block2_conv2_436899
identity˘$block2_conv1/StatefulPartitionedCall˘$block2_conv2/StatefulPartitionedCallˇ
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCallinput_2block2_conv1_436870block2_conv1_436872*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_block2_conv1_layer_call_and_return_conditional_losses_4368592&
$block2_conv1/StatefulPartitionedCallÝ
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_436897block2_conv2_436899*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_block2_conv2_layer_call_and_return_conditional_losses_4368862&
$block2_conv2/StatefulPartitionedCall
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_block2_pool_layer_call_and_return_conditional_losses_4368382
block2_pool/PartitionedCallĎ
IdentityIdentity$block2_pool/PartitionedCall:output:0%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙@::::2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall:X T
/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
!
_user_specified_name	input_2
Ž
´
?__inference_fog_layer_call_and_return_conditional_losses_437142
input_5

fog_437115

fog_437117

fog_437119

fog_437121
dense_1_437125
dense_1_437127
dense_2_437130
dense_2_437132
fog_output_437135
fog_output_437137
identity

identity_1˘dense_1/StatefulPartitionedCall˘dense_2/StatefulPartitionedCall˘fog/StatefulPartitionedCall˘"fog_output/StatefulPartitionedCallŚ
fog/StatefulPartitionedCallStatefulPartitionedCallinput_5
fog_437115
fog_437117
fog_437119
fog_437121*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *H
fCRA
?__inference_fog_layer_call_and_return_conditional_losses_4369652
fog/StatefulPartitionedCallű
flatten_1/PartitionedCallPartitionedCall$fog/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_4370212
flatten_1/PartitionedCallą
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1_437125dense_1_437127*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_4370402!
dense_1/StatefulPartitionedCallˇ
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_437130dense_2_437132*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_4370672!
dense_2/StatefulPartitionedCallĹ
"fog_output/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0fog_output_437135fog_output_437137*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_fog_output_layer_call_and_return_conditional_losses_4370942$
"fog_output/StatefulPartitionedCall
IdentityIdentity$fog/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^fog/StatefulPartitionedCall#^fog_output/StatefulPartitionedCall*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity

Identity_1Identity+fog_output/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^fog/StatefulPartitionedCall#^fog_output/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*V
_input_shapesE
C:˙˙˙˙˙˙˙˙˙@::::::::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2:
fog/StatefulPartitionedCallfog/StatefulPartitionedCall2H
"fog_output/StatefulPartitionedCall"fog_output/StatefulPartitionedCall:X T
/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
!
_user_specified_name	input_5
ç

?__inference_fog_layer_call_and_return_conditional_losses_436965

inputs
block2_conv1_436953
block2_conv1_436955
block2_conv2_436958
block2_conv2_436960
identity˘$block2_conv1/StatefulPartitionedCall˘$block2_conv2/StatefulPartitionedCallś
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCallinputsblock2_conv1_436953block2_conv1_436955*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_block2_conv1_layer_call_and_return_conditional_losses_4368592&
$block2_conv1/StatefulPartitionedCallÝ
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_436958block2_conv2_436960*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_block2_conv2_layer_call_and_return_conditional_losses_4368862&
$block2_conv2/StatefulPartitionedCall
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_block2_pool_layer_call_and_return_conditional_losses_4368382
block2_pool/PartitionedCallĎ
IdentityIdentity$block2_pool/PartitionedCall:output:0%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙@::::2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
Ť
ł
?__inference_fog_layer_call_and_return_conditional_losses_437175

inputs

fog_437148

fog_437150

fog_437152

fog_437154
dense_1_437158
dense_1_437160
dense_2_437163
dense_2_437165
fog_output_437168
fog_output_437170
identity

identity_1˘dense_1/StatefulPartitionedCall˘dense_2/StatefulPartitionedCall˘fog/StatefulPartitionedCall˘"fog_output/StatefulPartitionedCallĽ
fog/StatefulPartitionedCallStatefulPartitionedCallinputs
fog_437148
fog_437150
fog_437152
fog_437154*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *H
fCRA
?__inference_fog_layer_call_and_return_conditional_losses_4369372
fog/StatefulPartitionedCallű
flatten_1/PartitionedCallPartitionedCall$fog/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_4370212
flatten_1/PartitionedCallą
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1_437158dense_1_437160*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_4370402!
dense_1/StatefulPartitionedCallˇ
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_437163dense_2_437165*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_4370672!
dense_2/StatefulPartitionedCallĹ
"fog_output/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0fog_output_437168fog_output_437170*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_fog_output_layer_call_and_return_conditional_losses_4370942$
"fog_output/StatefulPartitionedCall
IdentityIdentity$fog/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^fog/StatefulPartitionedCall#^fog_output/StatefulPartitionedCall*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity

Identity_1Identity+fog_output/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^fog/StatefulPartitionedCall#^fog_output/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*V
_input_shapesE
C:˙˙˙˙˙˙˙˙˙@::::::::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2:
fog/StatefulPartitionedCallfog/StatefulPartitionedCall2H
"fog_output/StatefulPartitionedCall"fog_output/StatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
Ö


$__inference_fog_layer_call_fn_437399

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity

identity_1˘StatefulPartitionedCallů
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *C
_output_shapes1
/:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8 *H
fCRA
?__inference_fog_layer_call_and_return_conditional_losses_4371752
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*V
_input_shapesE
C:˙˙˙˙˙˙˙˙˙@::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
ă
}
(__inference_dense_2_layer_call_fn_437541

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallů
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_4370672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ą
Ť
C__inference_dense_1_layer_call_and_return_conditional_losses_437040

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
@*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙@:::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
Ĺ.
Ő
?__inference_fog_layer_call_and_return_conditional_losses_437372

inputs3
/fog_block2_conv1_conv2d_readvariableop_resource4
0fog_block2_conv1_biasadd_readvariableop_resource3
/fog_block2_conv2_conv2d_readvariableop_resource4
0fog_block2_conv2_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource-
)fog_output_matmul_readvariableop_resource.
*fog_output_biasadd_readvariableop_resource
identity

identity_1É
&fog/block2_conv1/Conv2D/ReadVariableOpReadVariableOp/fog_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02(
&fog/block2_conv1/Conv2D/ReadVariableOp×
fog/block2_conv1/Conv2DConv2Dinputs.fog/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
fog/block2_conv1/Conv2DŔ
'fog/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp0fog_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02)
'fog/block2_conv1/BiasAdd/ReadVariableOpÍ
fog/block2_conv1/BiasAddBiasAdd fog/block2_conv1/Conv2D:output:0/fog/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
fog/block2_conv1/BiasAdd
fog/block2_conv1/ReluRelu!fog/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
fog/block2_conv1/ReluĘ
&fog/block2_conv2/Conv2D/ReadVariableOpReadVariableOp/fog_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02(
&fog/block2_conv2/Conv2D/ReadVariableOpô
fog/block2_conv2/Conv2DConv2D#fog/block2_conv1/Relu:activations:0.fog/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
fog/block2_conv2/Conv2DŔ
'fog/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp0fog_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02)
'fog/block2_conv2/BiasAdd/ReadVariableOpÍ
fog/block2_conv2/BiasAddBiasAdd fog/block2_conv2/Conv2D:output:0/fog/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
fog/block2_conv2/BiasAdd
fog/block2_conv2/ReluRelu!fog/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
fog/block2_conv2/ReluĐ
fog/block2_pool/MaxPoolMaxPool#fog/block2_conv2/Relu:activations:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
ksize
*
paddingVALID*
strides
2
fog/block2_pool/MaxPools
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    2
flatten_1/Const 
flatten_1/ReshapeReshape fog/block2_pool/MaxPool:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
flatten_1/Reshape§
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
@*
dtype02
dense_1/MatMul/ReadVariableOp 
dense_1/MatMulMatMulflatten_1/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_1/MatMulĽ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp˘
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_1/Relu§
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_2/MatMul/ReadVariableOp 
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_2/MatMulĽ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp˘
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_2/BiasAddq
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_2/ReluŻ
 fog_output/MatMul/ReadVariableOpReadVariableOp)fog_output_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype02"
 fog_output/MatMul/ReadVariableOp¨
fog_output/MatMulMatMuldense_2/Relu:activations:0(fog_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
fog_output/MatMul­
!fog_output/BiasAdd/ReadVariableOpReadVariableOp*fog_output_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02#
!fog_output/BiasAdd/ReadVariableOp­
fog_output/BiasAddBiasAddfog_output/MatMul:product:0)fog_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
fog_output/BiasAdd
fog_output/SoftmaxSoftmaxfog_output/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
fog_output/Softmax}
IdentityIdentity fog/block2_pool/MaxPool:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identityt

Identity_1Identityfog_output/Softmax:softmax:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*V
_input_shapesE
C:˙˙˙˙˙˙˙˙˙@:::::::::::W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
ę

?__inference_fog_layer_call_and_return_conditional_losses_436919
input_2
block2_conv1_436907
block2_conv1_436909
block2_conv2_436912
block2_conv2_436914
identity˘$block2_conv1/StatefulPartitionedCall˘$block2_conv2/StatefulPartitionedCallˇ
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCallinput_2block2_conv1_436907block2_conv1_436909*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_block2_conv1_layer_call_and_return_conditional_losses_4368592&
$block2_conv1/StatefulPartitionedCallÝ
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_436912block2_conv2_436914*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_block2_conv2_layer_call_and_return_conditional_losses_4368862&
$block2_conv2/StatefulPartitionedCall
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_block2_pool_layer_call_and_return_conditional_losses_4368382
block2_pool/PartitionedCallĎ
IdentityIdentity$block2_pool/PartitionedCall:output:0%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙@::::2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall:X T
/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
!
_user_specified_name	input_2
ş

$__inference_fog_layer_call_fn_437490

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *H
fCRA
?__inference_fog_layer_call_and_return_conditional_losses_4369652
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
Ž
´
?__inference_fog_layer_call_and_return_conditional_losses_437112
input_5

fog_437006

fog_437008

fog_437010

fog_437012
dense_1_437051
dense_1_437053
dense_2_437078
dense_2_437080
fog_output_437105
fog_output_437107
identity

identity_1˘dense_1/StatefulPartitionedCall˘dense_2/StatefulPartitionedCall˘fog/StatefulPartitionedCall˘"fog_output/StatefulPartitionedCallŚ
fog/StatefulPartitionedCallStatefulPartitionedCallinput_5
fog_437006
fog_437008
fog_437010
fog_437012*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *H
fCRA
?__inference_fog_layer_call_and_return_conditional_losses_4369372
fog/StatefulPartitionedCallű
flatten_1/PartitionedCallPartitionedCall$fog/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_4370212
flatten_1/PartitionedCallą
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1_437051dense_1_437053*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_4370402!
dense_1/StatefulPartitionedCallˇ
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_437078dense_2_437080*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_4370672!
dense_2/StatefulPartitionedCallĹ
"fog_output/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0fog_output_437105fog_output_437107*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_fog_output_layer_call_and_return_conditional_losses_4370942$
"fog_output/StatefulPartitionedCall
IdentityIdentity$fog/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^fog/StatefulPartitionedCall#^fog_output/StatefulPartitionedCall*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity

Identity_1Identity+fog_output/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^fog/StatefulPartitionedCall#^fog_output/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*V
_input_shapesE
C:˙˙˙˙˙˙˙˙˙@::::::::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2:
fog/StatefulPartitionedCallfog/StatefulPartitionedCall2H
"fog_output/StatefulPartitionedCall"fog_output/StatefulPartitionedCall:X T
/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
!
_user_specified_name	input_5
ą
Ť
C__inference_dense_2_layer_call_and_return_conditional_losses_437532

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ď1
ŕ
!__inference__wrapped_model_436832
input_57
3fog_fog_block2_conv1_conv2d_readvariableop_resource8
4fog_fog_block2_conv1_biasadd_readvariableop_resource7
3fog_fog_block2_conv2_conv2d_readvariableop_resource8
4fog_fog_block2_conv2_biasadd_readvariableop_resource.
*fog_dense_1_matmul_readvariableop_resource/
+fog_dense_1_biasadd_readvariableop_resource.
*fog_dense_2_matmul_readvariableop_resource/
+fog_dense_2_biasadd_readvariableop_resource1
-fog_fog_output_matmul_readvariableop_resource2
.fog_fog_output_biasadd_readvariableop_resource
identity

identity_1Ő
*fog/fog/block2_conv1/Conv2D/ReadVariableOpReadVariableOp3fog_fog_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02,
*fog/fog/block2_conv1/Conv2D/ReadVariableOpä
fog/fog/block2_conv1/Conv2DConv2Dinput_52fog/fog/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
fog/fog/block2_conv1/Conv2DĚ
+fog/fog/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp4fog_fog_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+fog/fog/block2_conv1/BiasAdd/ReadVariableOpÝ
fog/fog/block2_conv1/BiasAddBiasAdd$fog/fog/block2_conv1/Conv2D:output:03fog/fog/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
fog/fog/block2_conv1/BiasAdd 
fog/fog/block2_conv1/ReluRelu%fog/fog/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
fog/fog/block2_conv1/ReluÖ
*fog/fog/block2_conv2/Conv2D/ReadVariableOpReadVariableOp3fog_fog_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02,
*fog/fog/block2_conv2/Conv2D/ReadVariableOp
fog/fog/block2_conv2/Conv2DConv2D'fog/fog/block2_conv1/Relu:activations:02fog/fog/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
fog/fog/block2_conv2/Conv2DĚ
+fog/fog/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp4fog_fog_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+fog/fog/block2_conv2/BiasAdd/ReadVariableOpÝ
fog/fog/block2_conv2/BiasAddBiasAdd$fog/fog/block2_conv2/Conv2D:output:03fog/fog/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
fog/fog/block2_conv2/BiasAdd 
fog/fog/block2_conv2/ReluRelu%fog/fog/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
fog/fog/block2_conv2/ReluÜ
fog/fog/block2_pool/MaxPoolMaxPool'fog/fog/block2_conv2/Relu:activations:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
ksize
*
paddingVALID*
strides
2
fog/fog/block2_pool/MaxPool{
fog/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    2
fog/flatten_1/Const°
fog/flatten_1/ReshapeReshape$fog/fog/block2_pool/MaxPool:output:0fog/flatten_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
fog/flatten_1/Reshapeł
!fog/dense_1/MatMul/ReadVariableOpReadVariableOp*fog_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
@*
dtype02#
!fog/dense_1/MatMul/ReadVariableOp°
fog/dense_1/MatMulMatMulfog/flatten_1/Reshape:output:0)fog/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
fog/dense_1/MatMulą
"fog/dense_1/BiasAdd/ReadVariableOpReadVariableOp+fog_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"fog/dense_1/BiasAdd/ReadVariableOp˛
fog/dense_1/BiasAddBiasAddfog/dense_1/MatMul:product:0*fog/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
fog/dense_1/BiasAdd}
fog/dense_1/ReluRelufog/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
fog/dense_1/Reluł
!fog/dense_2/MatMul/ReadVariableOpReadVariableOp*fog_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02#
!fog/dense_2/MatMul/ReadVariableOp°
fog/dense_2/MatMulMatMulfog/dense_1/Relu:activations:0)fog/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
fog/dense_2/MatMulą
"fog/dense_2/BiasAdd/ReadVariableOpReadVariableOp+fog_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02$
"fog/dense_2/BiasAdd/ReadVariableOp˛
fog/dense_2/BiasAddBiasAddfog/dense_2/MatMul:product:0*fog/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
fog/dense_2/BiasAdd}
fog/dense_2/ReluRelufog/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
fog/dense_2/Reluť
$fog/fog_output/MatMul/ReadVariableOpReadVariableOp-fog_fog_output_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype02&
$fog/fog_output/MatMul/ReadVariableOp¸
fog/fog_output/MatMulMatMulfog/dense_2/Relu:activations:0,fog/fog_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
fog/fog_output/MatMulš
%fog/fog_output/BiasAdd/ReadVariableOpReadVariableOp.fog_fog_output_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02'
%fog/fog_output/BiasAdd/ReadVariableOp˝
fog/fog_output/BiasAddBiasAddfog/fog_output/MatMul:product:0-fog/fog_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
fog/fog_output/BiasAdd
fog/fog_output/SoftmaxSoftmaxfog/fog_output/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
fog/fog_output/Softmax
IdentityIdentity$fog/fog/block2_pool/MaxPool:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identityx

Identity_1Identity fog/fog_output/Softmax:softmax:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*V
_input_shapesE
C:˙˙˙˙˙˙˙˙˙@:::::::::::X T
/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
!
_user_specified_name	input_5
ş

$__inference_fog_layer_call_fn_437477

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *H
fCRA
?__inference_fog_layer_call_and_return_conditional_losses_4369372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
ż
a
E__inference_flatten_1_layer_call_and_return_conditional_losses_437021

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙@2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙@2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ç

?__inference_fog_layer_call_and_return_conditional_losses_436937

inputs
block2_conv1_436925
block2_conv1_436927
block2_conv2_436930
block2_conv2_436932
identity˘$block2_conv1/StatefulPartitionedCall˘$block2_conv2/StatefulPartitionedCallś
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCallinputsblock2_conv1_436925block2_conv1_436927*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_block2_conv1_layer_call_and_return_conditional_losses_4368592&
$block2_conv1/StatefulPartitionedCallÝ
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_436930block2_conv2_436932*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_block2_conv2_layer_call_and_return_conditional_losses_4368862&
$block2_conv2/StatefulPartitionedCall
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *P
fKRI
G__inference_block2_pool_layer_call_and_return_conditional_losses_4368382
block2_pool/PartitionedCallĎ
IdentityIdentity$block2_pool/PartitionedCall:output:0%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙@::::2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
Ť
ł
?__inference_fog_layer_call_and_return_conditional_losses_437232

inputs

fog_437205

fog_437207

fog_437209

fog_437211
dense_1_437215
dense_1_437217
dense_2_437220
dense_2_437222
fog_output_437225
fog_output_437227
identity

identity_1˘dense_1/StatefulPartitionedCall˘dense_2/StatefulPartitionedCall˘fog/StatefulPartitionedCall˘"fog_output/StatefulPartitionedCallĽ
fog/StatefulPartitionedCallStatefulPartitionedCallinputs
fog_437205
fog_437207
fog_437209
fog_437211*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *H
fCRA
?__inference_fog_layer_call_and_return_conditional_losses_4369652
fog/StatefulPartitionedCallű
flatten_1/PartitionedCallPartitionedCall$fog/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *N
fIRG
E__inference_flatten_1_layer_call_and_return_conditional_losses_4370212
flatten_1/PartitionedCallą
dense_1/StatefulPartitionedCallStatefulPartitionedCall"flatten_1/PartitionedCall:output:0dense_1_437215dense_1_437217*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_4370402!
dense_1/StatefulPartitionedCallˇ
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_437220dense_2_437222*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_4370672!
dense_2/StatefulPartitionedCallĹ
"fog_output/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0fog_output_437225fog_output_437227*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_fog_output_layer_call_and_return_conditional_losses_4370942$
"fog_output/StatefulPartitionedCall
IdentityIdentity$fog/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^fog/StatefulPartitionedCall#^fog_output/StatefulPartitionedCall*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity

Identity_1Identity+fog_output/StatefulPartitionedCall:output:0 ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^fog/StatefulPartitionedCall#^fog_output/StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*V
_input_shapesE
C:˙˙˙˙˙˙˙˙˙@::::::::::2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2:
fog/StatefulPartitionedCallfog/StatefulPartitionedCall2H
"fog_output/StatefulPartitionedCall"fog_output/StatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
	
°
H__inference_block2_conv1_layer_call_and_return_conditional_losses_436859

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙@:::W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
	
°
H__inference_block2_conv1_layer_call_and_return_conditional_losses_437572

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙@:::W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
˝

$__inference_fog_layer_call_fn_436948
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *H
fCRA
?__inference_fog_layer_call_and_return_conditional_losses_4369372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙@::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
!
_user_specified_name	input_2


-__inference_block2_conv1_layer_call_fn_437581

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_block2_conv1_layer_call_and_return_conditional_losses_4368592
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:˙˙˙˙˙˙˙˙˙@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
ś
Ž
F__inference_fog_output_layer_call_and_return_conditional_losses_437552

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ů


$__inference_fog_layer_call_fn_437257
input_5
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity

identity_1˘StatefulPartitionedCallú
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *C
_output_shapes1
/:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8 *H
fCRA
?__inference_fog_layer_call_and_return_conditional_losses_4372322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*V
_input_shapesE
C:˙˙˙˙˙˙˙˙˙@::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
!
_user_specified_name	input_5
Ö


$__inference_fog_layer_call_fn_437426

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity

identity_1˘StatefulPartitionedCallů
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *C
_output_shapes1
/:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8 *H
fCRA
?__inference_fog_layer_call_and_return_conditional_losses_4372322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*V
_input_shapesE
C:˙˙˙˙˙˙˙˙˙@::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
Ĺ.
Ő
?__inference_fog_layer_call_and_return_conditional_losses_437329

inputs3
/fog_block2_conv1_conv2d_readvariableop_resource4
0fog_block2_conv1_biasadd_readvariableop_resource3
/fog_block2_conv2_conv2d_readvariableop_resource4
0fog_block2_conv2_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource-
)fog_output_matmul_readvariableop_resource.
*fog_output_biasadd_readvariableop_resource
identity

identity_1É
&fog/block2_conv1/Conv2D/ReadVariableOpReadVariableOp/fog_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02(
&fog/block2_conv1/Conv2D/ReadVariableOp×
fog/block2_conv1/Conv2DConv2Dinputs.fog/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
fog/block2_conv1/Conv2DŔ
'fog/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp0fog_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02)
'fog/block2_conv1/BiasAdd/ReadVariableOpÍ
fog/block2_conv1/BiasAddBiasAdd fog/block2_conv1/Conv2D:output:0/fog/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
fog/block2_conv1/BiasAdd
fog/block2_conv1/ReluRelu!fog/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
fog/block2_conv1/ReluĘ
&fog/block2_conv2/Conv2D/ReadVariableOpReadVariableOp/fog_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02(
&fog/block2_conv2/Conv2D/ReadVariableOpô
fog/block2_conv2/Conv2DConv2D#fog/block2_conv1/Relu:activations:0.fog/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
fog/block2_conv2/Conv2DŔ
'fog/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp0fog_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02)
'fog/block2_conv2/BiasAdd/ReadVariableOpÍ
fog/block2_conv2/BiasAddBiasAdd fog/block2_conv2/Conv2D:output:0/fog/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
fog/block2_conv2/BiasAdd
fog/block2_conv2/ReluRelu!fog/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
fog/block2_conv2/ReluĐ
fog/block2_pool/MaxPoolMaxPool#fog/block2_conv2/Relu:activations:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
ksize
*
paddingVALID*
strides
2
fog/block2_pool/MaxPools
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"˙˙˙˙    2
flatten_1/Const 
flatten_1/ReshapeReshape fog/block2_pool/MaxPool:output:0flatten_1/Const:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙@2
flatten_1/Reshape§
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
@*
dtype02
dense_1/MatMul/ReadVariableOp 
dense_1/MatMulMatMulflatten_1/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_1/MatMulĽ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp˘
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_1/Relu§
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_2/MatMul/ReadVariableOp 
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_2/MatMulĽ
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp˘
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_2/BiasAddq
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
dense_2/ReluŻ
 fog_output/MatMul/ReadVariableOpReadVariableOp)fog_output_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype02"
 fog_output/MatMul/ReadVariableOp¨
fog_output/MatMulMatMuldense_2/Relu:activations:0(fog_output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
fog_output/MatMul­
!fog_output/BiasAdd/ReadVariableOpReadVariableOp*fog_output_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02#
!fog_output/BiasAdd/ReadVariableOp­
fog_output/BiasAddBiasAddfog_output/MatMul:product:0)fog_output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
fog_output/BiasAdd
fog_output/SoftmaxSoftmaxfog_output/BiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
fog_output/Softmax}
IdentityIdentity fog/block2_pool/MaxPool:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identityt

Identity_1Identityfog_output/Softmax:softmax:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*V
_input_shapesE
C:˙˙˙˙˙˙˙˙˙@:::::::::::W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
Ű
¤
?__inference_fog_layer_call_and_return_conditional_losses_437464

inputs/
+block2_conv1_conv2d_readvariableop_resource0
,block2_conv1_biasadd_readvariableop_resource/
+block2_conv2_conv2d_readvariableop_resource0
,block2_conv2_biasadd_readvariableop_resource
identity˝
"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02$
"block2_conv1/Conv2D/ReadVariableOpË
block2_conv1/Conv2DConv2Dinputs*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
block2_conv1/Conv2D´
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block2_conv1/BiasAdd/ReadVariableOp˝
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
block2_conv1/BiasAdd
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
block2_conv1/Reluž
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block2_conv2/Conv2D/ReadVariableOpä
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
block2_conv2/Conv2D´
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block2_conv2/BiasAdd/ReadVariableOp˝
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
block2_conv2/BiasAdd
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
block2_conv2/ReluÄ
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
ksize
*
paddingVALID*
strides
2
block2_pool/MaxPooly
IdentityIdentityblock2_pool/MaxPool:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙@:::::W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
ă
}
(__inference_dense_1_layer_call_fn_437521

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCallů
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_4370402
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙@::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
	
°
H__inference_block2_conv2_layer_call_and_return_conditional_losses_437592

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:˙˙˙˙˙˙˙˙˙:::X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
"
Ň
__inference__traced_save_437655
file_prefix-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop0
,savev2_fog_output_kernel_read_readvariableop.
*savev2_fog_output_bias_read_readvariableop2
.savev2_block2_conv1_kernel_read_readvariableop0
,savev2_block2_conv1_bias_read_readvariableop2
.savev2_block2_conv2_kernel_read_readvariableop0
,savev2_block2_conv2_bias_read_readvariableop
savev2_const

identity_1˘MergeV2Checkpoints
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
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_25df9c4bb44440d9a40afbe872d80fdd/part2	
Const_1
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
ShardedFilename/shardŚ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop,savev2_fog_output_kernel_read_readvariableop*savev2_fog_output_bias_read_readvariableop.savev2_block2_conv1_kernel_read_readvariableop,savev2_block2_conv1_bias_read_readvariableop.savev2_block2_conv2_kernel_read_readvariableop,savev2_block2_conv2_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2ş
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesĄ
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

identity_1Identity_1:output:0*
_input_shapesr
p: :
@::
::	
:
:@:::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
@:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	
: 

_output_shapes
:
:-)
'
_output_shapes
:@:!

_output_shapes	
::.	*
(
_output_shapes
::!


_output_shapes	
::

_output_shapes
: 
ą
Ť
C__inference_dense_2_layer_call_and_return_conditional_losses_437067

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs


-__inference_block2_conv2_layer_call_fn_437601

inputs
unknown
	unknown_0
identity˘StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:˙˙˙˙˙˙˙˙˙*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *Q
fLRJ
H__inference_block2_conv2_layer_call_and_return_conditional_losses_4368862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:˙˙˙˙˙˙˙˙˙::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
ť


$__inference_signature_wrapper_437286
input_5
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity

identity_1˘StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *C
_output_shapes1
/:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
*,
_read_only_resource_inputs

	
*2
config_proto" 

CPU

GPU2 *0J 8 **
f%R#
!__inference__wrapped_model_4368322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*V
_input_shapesE
C:˙˙˙˙˙˙˙˙˙@::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
!
_user_specified_name	input_5
ý
c
G__inference_block2_pool_layer_call_and_return_conditional_losses_436838

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙:r n
J
_output_shapes8
6:4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs
Ű
¤
?__inference_fog_layer_call_and_return_conditional_losses_437445

inputs/
+block2_conv1_conv2d_readvariableop_resource0
,block2_conv1_biasadd_readvariableop_resource/
+block2_conv2_conv2d_readvariableop_resource0
,block2_conv2_biasadd_readvariableop_resource
identity˝
"block2_conv1/Conv2D/ReadVariableOpReadVariableOp+block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02$
"block2_conv1/Conv2D/ReadVariableOpË
block2_conv1/Conv2DConv2Dinputs*block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
block2_conv1/Conv2D´
#block2_conv1/BiasAdd/ReadVariableOpReadVariableOp,block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block2_conv1/BiasAdd/ReadVariableOp˝
block2_conv1/BiasAddBiasAddblock2_conv1/Conv2D:output:0+block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
block2_conv1/BiasAdd
block2_conv1/ReluRelublock2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
block2_conv1/Reluž
"block2_conv2/Conv2D/ReadVariableOpReadVariableOp+block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02$
"block2_conv2/Conv2D/ReadVariableOpä
block2_conv2/Conv2DConv2Dblock2_conv1/Relu:activations:0*block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
paddingSAME*
strides
2
block2_conv2/Conv2D´
#block2_conv2/BiasAdd/ReadVariableOpReadVariableOp,block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02%
#block2_conv2/BiasAdd/ReadVariableOp˝
block2_conv2/BiasAddBiasAddblock2_conv2/Conv2D:output:0+block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
block2_conv2/BiasAdd
block2_conv2/ReluRelublock2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2
block2_conv2/ReluÄ
block2_pool/MaxPoolMaxPoolblock2_conv2/Relu:activations:0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙*
ksize
*
paddingVALID*
strides
2
block2_pool/MaxPooly
IdentityIdentityblock2_pool/MaxPool:output:0*
T0*0
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:˙˙˙˙˙˙˙˙˙@:::::W S
/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
ą
Ť
C__inference_dense_1_layer_call_and_return_conditional_losses_437512

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
@*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙@:::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙@
 
_user_specified_nameinputs
ś
Ž
F__inference_fog_output_layer_call_and_return_conditional_losses_437094

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
2

Identity"
identityIdentity:output:0*/
_input_shapes
:˙˙˙˙˙˙˙˙˙:::P L
(
_output_shapes
:˙˙˙˙˙˙˙˙˙
 
_user_specified_nameinputs"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*÷
serving_defaultă
C
input_58
serving_default_input_5:0˙˙˙˙˙˙˙˙˙@@
fog9
StatefulPartitionedCall:0˙˙˙˙˙˙˙˙˙>

fog_output0
StatefulPartitionedCall:1˙˙˙˙˙˙˙˙˙
tensorflow/serving/predict:ł
ĆI
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
regularization_losses
trainable_variables
		variables

	keras_api

signatures
g_default_save_signature
*h&call_and_return_all_conditional_losses
i__call__"ÔF
_tf_keras_network¸F{"class_name": "Functional", "name": "fog", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "fog", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}, "name": "input_5", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "fog", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "block2_conv1", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv1", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv2", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv2", "inbound_nodes": [[["block2_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block2_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block2_pool", "inbound_nodes": [[["block2_conv2", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["block2_pool", 0, 0]]}, "name": "fog", "inbound_nodes": [[["input_5", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["fog", 1, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["flatten_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fog_output", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fog_output", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}], "input_layers": [["input_5", 0, 0]], "output_layers": [["fog", 1, 0], ["fog_output", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 64]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "fog", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}, "name": "input_5", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "fog", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "block2_conv1", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv1", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv2", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv2", "inbound_nodes": [[["block2_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block2_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block2_pool", "inbound_nodes": [[["block2_conv2", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["block2_pool", 0, 0]]}, "name": "fog", "inbound_nodes": [[["input_5", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten_1", "inbound_nodes": [[["fog", 1, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["flatten_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "fog_output", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "fog_output", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}], "input_layers": [["input_5", 0, 0]], "output_layers": [["fog", 1, 0], ["fog_output", 0, 0]]}}}
ű"ř
_tf_keras_input_layerŘ{"class_name": "InputLayer", "name": "input_5", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 64]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}}
ş'
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
regularization_losses
trainable_variables
	variables
	keras_api
*j&call_and_return_all_conditional_losses
k__call__"Ă%
_tf_keras_network§%{"class_name": "Functional", "name": "fog", "trainable": false, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "fog", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "block2_conv1", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv1", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv2", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv2", "inbound_nodes": [[["block2_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block2_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block2_pool", "inbound_nodes": [[["block2_conv2", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["block2_pool", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 64]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "fog", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "block2_conv1", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv1", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "block2_conv2", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "block2_conv2", "inbound_nodes": [[["block2_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "block2_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "block2_pool", "inbound_nodes": [[["block2_conv2", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["block2_pool", 0, 0]]}}}
ć
regularization_losses
trainable_variables
	variables
	keras_api
*l&call_and_return_all_conditional_losses
m__call__"×
_tf_keras_layer˝{"class_name": "Flatten", "name": "flatten_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_1", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
ő

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
*n&call_and_return_all_conditional_losses
o__call__"Đ
_tf_keras_layerś{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8192}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8192]}}
ó

kernel
bias
 regularization_losses
!trainable_variables
"	variables
#	keras_api
*p&call_and_return_all_conditional_losses
q__call__"Î
_tf_keras_layer´{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
ű

$kernel
%bias
&regularization_losses
'trainable_variables
(	variables
)	keras_api
*r&call_and_return_all_conditional_losses
s__call__"Ö
_tf_keras_layerź{"class_name": "Dense", "name": "fog_output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "fog_output", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
 "
trackable_list_wrapper
J
0
1
2
3
$4
%5"
trackable_list_wrapper
f
*0
+1
,2
-3
4
5
6
7
$8
%9"
trackable_list_wrapper
Ę
regularization_losses

.layers
/non_trainable_variables
0layer_regularization_losses
1metrics
2layer_metrics
trainable_variables
		variables
i__call__
g_default_save_signature
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
,
tserving_default"
signature_map
ű"ř
_tf_keras_input_layerŘ{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 64]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16, 16, 64]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
ý	

*kernel
+bias
3regularization_losses
4trainable_variables
5	variables
6	keras_api
*u&call_and_return_all_conditional_losses
v__call__"Ř
_tf_keras_layerž{"class_name": "Conv2D", "name": "block2_conv1", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block2_conv1", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 64]}}
˙	

,kernel
-bias
7regularization_losses
8trainable_variables
9	variables
:	keras_api
*w&call_and_return_all_conditional_losses
x__call__"Ú
_tf_keras_layerŔ{"class_name": "Conv2D", "name": "block2_conv2", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block2_conv2", "trainable": false, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 128]}}
ů
;regularization_losses
<trainable_variables
=	variables
>	keras_api
*y&call_and_return_all_conditional_losses
z__call__"ę
_tf_keras_layerĐ{"class_name": "MaxPooling2D", "name": "block2_pool", "trainable": false, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "block2_pool", "trainable": false, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
*0
+1
,2
-3"
trackable_list_wrapper
­
regularization_losses

?layers
@non_trainable_variables
Alayer_regularization_losses
Bmetrics
Clayer_metrics
trainable_variables
	variables
k__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
regularization_losses

Dlayers
Enon_trainable_variables
Flayer_regularization_losses
Gmetrics
Hlayer_metrics
trainable_variables
	variables
m__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
": 
@2dense_1/kernel
:2dense_1/bias
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
­
regularization_losses

Ilayers
Jnon_trainable_variables
Klayer_regularization_losses
Lmetrics
Mlayer_metrics
trainable_variables
	variables
o__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
": 
2dense_2/kernel
:2dense_2/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
 regularization_losses

Nlayers
Onon_trainable_variables
Player_regularization_losses
Qmetrics
Rlayer_metrics
!trainable_variables
"	variables
q__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
$:"	
2fog_output/kernel
:
2fog_output/bias
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
­
&regularization_losses

Slayers
Tnon_trainable_variables
Ulayer_regularization_losses
Vmetrics
Wlayer_metrics
'trainable_variables
(	variables
s__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
.:,@2block2_conv1/kernel
 :2block2_conv1/bias
/:-2block2_conv2/kernel
 :2block2_conv2/bias
J
0
1
2
3
4
5"
trackable_list_wrapper
<
*0
+1
,2
-3"
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
.
*0
+1"
trackable_list_wrapper
­
3regularization_losses

Xlayers
Ynon_trainable_variables
Zlayer_regularization_losses
[metrics
\layer_metrics
4trainable_variables
5	variables
v__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
­
7regularization_losses

]layers
^non_trainable_variables
_layer_regularization_losses
`metrics
alayer_metrics
8trainable_variables
9	variables
x__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
;regularization_losses

blayers
cnon_trainable_variables
dlayer_regularization_losses
emetrics
flayer_metrics
<trainable_variables
=	variables
z__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
<
0
1
2
3"
trackable_list_wrapper
<
*0
+1
,2
-3"
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
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
,0
-1"
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
ç2ä
!__inference__wrapped_model_436832ž
˛
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *.˘+
)&
input_5˙˙˙˙˙˙˙˙˙@
Ę2Ç
?__inference_fog_layer_call_and_return_conditional_losses_437142
?__inference_fog_layer_call_and_return_conditional_losses_437329
?__inference_fog_layer_call_and_return_conditional_losses_437372
?__inference_fog_layer_call_and_return_conditional_losses_437112Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Ţ2Ű
$__inference_fog_layer_call_fn_437426
$__inference_fog_layer_call_fn_437257
$__inference_fog_layer_call_fn_437399
$__inference_fog_layer_call_fn_437200Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Ę2Ç
?__inference_fog_layer_call_and_return_conditional_losses_437464
?__inference_fog_layer_call_and_return_conditional_losses_437445
?__inference_fog_layer_call_and_return_conditional_losses_436904
?__inference_fog_layer_call_and_return_conditional_losses_436919Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
Ţ2Ű
$__inference_fog_layer_call_fn_437477
$__inference_fog_layer_call_fn_436948
$__inference_fog_layer_call_fn_437490
$__inference_fog_layer_call_fn_436976Ŕ
ˇ˛ł
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsŞ 
annotationsŞ *
 
ď2ě
E__inference_flatten_1_layer_call_and_return_conditional_losses_437496˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ô2Ń
*__inference_flatten_1_layer_call_fn_437501˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
í2ę
C__inference_dense_1_layer_call_and_return_conditional_losses_437512˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ň2Ď
(__inference_dense_1_layer_call_fn_437521˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
í2ę
C__inference_dense_2_layer_call_and_return_conditional_losses_437532˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ň2Ď
(__inference_dense_2_layer_call_fn_437541˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
đ2í
F__inference_fog_output_layer_call_and_return_conditional_losses_437552˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ő2Ň
+__inference_fog_output_layer_call_fn_437561˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
3B1
$__inference_signature_wrapper_437286input_5
ň2ď
H__inference_block2_conv1_layer_call_and_return_conditional_losses_437572˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
×2Ô
-__inference_block2_conv1_layer_call_fn_437581˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
ň2ď
H__inference_block2_conv2_layer_call_and_return_conditional_losses_437592˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
×2Ô
-__inference_block2_conv2_layer_call_fn_437601˘
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *
 
Ż2Ź
G__inference_block2_pool_layer_call_and_return_conditional_losses_436838ŕ
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *@˘=
;84˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
2
,__inference_block2_pool_layer_call_fn_436844ŕ
˛
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŞ *@˘=
;84˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ô
!__inference__wrapped_model_436832Ž
*+,-$%8˘5
.˘+
)&
input_5˙˙˙˙˙˙˙˙˙@
Ş "fŞc
-
fog&#
fog˙˙˙˙˙˙˙˙˙
2

fog_output$!

fog_output˙˙˙˙˙˙˙˙˙
š
H__inference_block2_conv1_layer_call_and_return_conditional_losses_437572m*+7˘4
-˘*
(%
inputs˙˙˙˙˙˙˙˙˙@
Ş ".˘+
$!
0˙˙˙˙˙˙˙˙˙
 
-__inference_block2_conv1_layer_call_fn_437581`*+7˘4
-˘*
(%
inputs˙˙˙˙˙˙˙˙˙@
Ş "!˙˙˙˙˙˙˙˙˙ş
H__inference_block2_conv2_layer_call_and_return_conditional_losses_437592n,-8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙
Ş ".˘+
$!
0˙˙˙˙˙˙˙˙˙
 
-__inference_block2_conv2_layer_call_fn_437601a,-8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙
Ş "!˙˙˙˙˙˙˙˙˙ę
G__inference_block2_pool_layer_call_and_return_conditional_losses_436838R˘O
H˘E
C@
inputs4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş "H˘E
>;
04˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
 Â
,__inference_block2_pool_layer_call_fn_436844R˘O
H˘E
C@
inputs4˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙
Ş ";84˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙˙Ľ
C__inference_dense_1_layer_call_and_return_conditional_losses_437512^0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙@
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 }
(__inference_dense_1_layer_call_fn_437521Q0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙@
Ş "˙˙˙˙˙˙˙˙˙Ľ
C__inference_dense_2_layer_call_and_return_conditional_losses_437532^0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "&˘#

0˙˙˙˙˙˙˙˙˙
 }
(__inference_dense_2_layer_call_fn_437541Q0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙Ť
E__inference_flatten_1_layer_call_and_return_conditional_losses_437496b8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙
Ş "&˘#

0˙˙˙˙˙˙˙˙˙@
 
*__inference_flatten_1_layer_call_fn_437501U8˘5
.˘+
)&
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙@ť
?__inference_fog_layer_call_and_return_conditional_losses_436904x*+,-@˘=
6˘3
)&
input_2˙˙˙˙˙˙˙˙˙@
p

 
Ş ".˘+
$!
0˙˙˙˙˙˙˙˙˙
 ť
?__inference_fog_layer_call_and_return_conditional_losses_436919x*+,-@˘=
6˘3
)&
input_2˙˙˙˙˙˙˙˙˙@
p 

 
Ş ".˘+
$!
0˙˙˙˙˙˙˙˙˙
 č
?__inference_fog_layer_call_and_return_conditional_losses_437112¤
*+,-$%@˘=
6˘3
)&
input_5˙˙˙˙˙˙˙˙˙@
p

 
Ş "T˘Q
JG
&#
0/0˙˙˙˙˙˙˙˙˙

0/1˙˙˙˙˙˙˙˙˙

 č
?__inference_fog_layer_call_and_return_conditional_losses_437142¤
*+,-$%@˘=
6˘3
)&
input_5˙˙˙˙˙˙˙˙˙@
p 

 
Ş "T˘Q
JG
&#
0/0˙˙˙˙˙˙˙˙˙

0/1˙˙˙˙˙˙˙˙˙

 ç
?__inference_fog_layer_call_and_return_conditional_losses_437329Ł
*+,-$%?˘<
5˘2
(%
inputs˙˙˙˙˙˙˙˙˙@
p

 
Ş "T˘Q
JG
&#
0/0˙˙˙˙˙˙˙˙˙

0/1˙˙˙˙˙˙˙˙˙

 ç
?__inference_fog_layer_call_and_return_conditional_losses_437372Ł
*+,-$%?˘<
5˘2
(%
inputs˙˙˙˙˙˙˙˙˙@
p 

 
Ş "T˘Q
JG
&#
0/0˙˙˙˙˙˙˙˙˙

0/1˙˙˙˙˙˙˙˙˙

 ş
?__inference_fog_layer_call_and_return_conditional_losses_437445w*+,-?˘<
5˘2
(%
inputs˙˙˙˙˙˙˙˙˙@
p

 
Ş ".˘+
$!
0˙˙˙˙˙˙˙˙˙
 ş
?__inference_fog_layer_call_and_return_conditional_losses_437464w*+,-?˘<
5˘2
(%
inputs˙˙˙˙˙˙˙˙˙@
p 

 
Ş ".˘+
$!
0˙˙˙˙˙˙˙˙˙
 
$__inference_fog_layer_call_fn_436948k*+,-@˘=
6˘3
)&
input_2˙˙˙˙˙˙˙˙˙@
p

 
Ş "!˙˙˙˙˙˙˙˙˙
$__inference_fog_layer_call_fn_436976k*+,-@˘=
6˘3
)&
input_2˙˙˙˙˙˙˙˙˙@
p 

 
Ş "!˙˙˙˙˙˙˙˙˙ż
$__inference_fog_layer_call_fn_437200
*+,-$%@˘=
6˘3
)&
input_5˙˙˙˙˙˙˙˙˙@
p

 
Ş "FC
$!
0˙˙˙˙˙˙˙˙˙

1˙˙˙˙˙˙˙˙˙
ż
$__inference_fog_layer_call_fn_437257
*+,-$%@˘=
6˘3
)&
input_5˙˙˙˙˙˙˙˙˙@
p 

 
Ş "FC
$!
0˙˙˙˙˙˙˙˙˙

1˙˙˙˙˙˙˙˙˙
ž
$__inference_fog_layer_call_fn_437399
*+,-$%?˘<
5˘2
(%
inputs˙˙˙˙˙˙˙˙˙@
p

 
Ş "FC
$!
0˙˙˙˙˙˙˙˙˙

1˙˙˙˙˙˙˙˙˙
ž
$__inference_fog_layer_call_fn_437426
*+,-$%?˘<
5˘2
(%
inputs˙˙˙˙˙˙˙˙˙@
p 

 
Ş "FC
$!
0˙˙˙˙˙˙˙˙˙

1˙˙˙˙˙˙˙˙˙

$__inference_fog_layer_call_fn_437477j*+,-?˘<
5˘2
(%
inputs˙˙˙˙˙˙˙˙˙@
p

 
Ş "!˙˙˙˙˙˙˙˙˙
$__inference_fog_layer_call_fn_437490j*+,-?˘<
5˘2
(%
inputs˙˙˙˙˙˙˙˙˙@
p 

 
Ş "!˙˙˙˙˙˙˙˙˙§
F__inference_fog_output_layer_call_and_return_conditional_losses_437552]$%0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "%˘"

0˙˙˙˙˙˙˙˙˙

 
+__inference_fog_output_layer_call_fn_437561P$%0˘-
&˘#
!
inputs˙˙˙˙˙˙˙˙˙
Ş "˙˙˙˙˙˙˙˙˙
â
$__inference_signature_wrapper_437286š
*+,-$%C˘@
˘ 
9Ş6
4
input_5)&
input_5˙˙˙˙˙˙˙˙˙@"fŞc
-
fog&#
fog˙˙˙˙˙˙˙˙˙
2

fog_output$!

fog_output˙˙˙˙˙˙˙˙˙
