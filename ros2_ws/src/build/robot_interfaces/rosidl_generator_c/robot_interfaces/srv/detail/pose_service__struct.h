// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from robot_interfaces:srv/PoseService.idl
// generated code does not contain a copyright notice

#ifndef ROBOT_INTERFACES__SRV__DETAIL__POSE_SERVICE__STRUCT_H_
#define ROBOT_INTERFACES__SRV__DETAIL__POSE_SERVICE__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'request'
#include "std_msgs/msg/detail/string__struct.h"

/// Struct defined in srv/PoseService in the package robot_interfaces.
typedef struct robot_interfaces__srv__PoseService_Request
{
  std_msgs__msg__String request;
} robot_interfaces__srv__PoseService_Request;

// Struct for a sequence of robot_interfaces__srv__PoseService_Request.
typedef struct robot_interfaces__srv__PoseService_Request__Sequence
{
  robot_interfaces__srv__PoseService_Request * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} robot_interfaces__srv__PoseService_Request__Sequence;


// Constants defined in the message

// Include directives for member types
// Member 'response'
#include "robot_interfaces/msg/detail/ego_pose__struct.h"

/// Struct defined in srv/PoseService in the package robot_interfaces.
typedef struct robot_interfaces__srv__PoseService_Response
{
  robot_interfaces__msg__EgoPose response;
} robot_interfaces__srv__PoseService_Response;

// Struct for a sequence of robot_interfaces__srv__PoseService_Response.
typedef struct robot_interfaces__srv__PoseService_Response__Sequence
{
  robot_interfaces__srv__PoseService_Response * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} robot_interfaces__srv__PoseService_Response__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // ROBOT_INTERFACES__SRV__DETAIL__POSE_SERVICE__STRUCT_H_
