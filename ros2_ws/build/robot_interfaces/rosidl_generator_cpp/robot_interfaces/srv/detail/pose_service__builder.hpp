// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from robot_interfaces:srv/PoseService.idl
// generated code does not contain a copyright notice

#ifndef ROBOT_INTERFACES__SRV__DETAIL__POSE_SERVICE__BUILDER_HPP_
#define ROBOT_INTERFACES__SRV__DETAIL__POSE_SERVICE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "robot_interfaces/srv/detail/pose_service__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace robot_interfaces
{

namespace srv
{

namespace builder
{

class Init_PoseService_Request_request
{
public:
  Init_PoseService_Request_request()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::robot_interfaces::srv::PoseService_Request request(::robot_interfaces::srv::PoseService_Request::_request_type arg)
  {
    msg_.request = std::move(arg);
    return std::move(msg_);
  }

private:
  ::robot_interfaces::srv::PoseService_Request msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::robot_interfaces::srv::PoseService_Request>()
{
  return robot_interfaces::srv::builder::Init_PoseService_Request_request();
}

}  // namespace robot_interfaces


namespace robot_interfaces
{

namespace srv
{

namespace builder
{

class Init_PoseService_Response_response
{
public:
  Init_PoseService_Response_response()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  ::robot_interfaces::srv::PoseService_Response response(::robot_interfaces::srv::PoseService_Response::_response_type arg)
  {
    msg_.response = std::move(arg);
    return std::move(msg_);
  }

private:
  ::robot_interfaces::srv::PoseService_Response msg_;
};

}  // namespace builder

}  // namespace srv

template<typename MessageType>
auto build();

template<>
inline
auto build<::robot_interfaces::srv::PoseService_Response>()
{
  return robot_interfaces::srv::builder::Init_PoseService_Response_response();
}

}  // namespace robot_interfaces

#endif  // ROBOT_INTERFACES__SRV__DETAIL__POSE_SERVICE__BUILDER_HPP_
