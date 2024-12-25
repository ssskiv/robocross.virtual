// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from robot_interfaces:msg/EgoPose.idl
// generated code does not contain a copyright notice

#ifndef ROBOT_INTERFACES__MSG__DETAIL__EGO_POSE__BUILDER_HPP_
#define ROBOT_INTERFACES__MSG__DETAIL__EGO_POSE__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "robot_interfaces/msg/detail/ego_pose__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace robot_interfaces
{

namespace msg
{

namespace builder
{

class Init_EgoPose_orientation
{
public:
  explicit Init_EgoPose_orientation(::robot_interfaces::msg::EgoPose & msg)
  : msg_(msg)
  {}
  ::robot_interfaces::msg::EgoPose orientation(::robot_interfaces::msg::EgoPose::_orientation_type arg)
  {
    msg_.orientation = std::move(arg);
    return std::move(msg_);
  }

private:
  ::robot_interfaces::msg::EgoPose msg_;
};

class Init_EgoPose_lon
{
public:
  explicit Init_EgoPose_lon(::robot_interfaces::msg::EgoPose & msg)
  : msg_(msg)
  {}
  Init_EgoPose_orientation lon(::robot_interfaces::msg::EgoPose::_lon_type arg)
  {
    msg_.lon = std::move(arg);
    return Init_EgoPose_orientation(msg_);
  }

private:
  ::robot_interfaces::msg::EgoPose msg_;
};

class Init_EgoPose_lat
{
public:
  Init_EgoPose_lat()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_EgoPose_lon lat(::robot_interfaces::msg::EgoPose::_lat_type arg)
  {
    msg_.lat = std::move(arg);
    return Init_EgoPose_lon(msg_);
  }

private:
  ::robot_interfaces::msg::EgoPose msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::robot_interfaces::msg::EgoPose>()
{
  return robot_interfaces::msg::builder::Init_EgoPose_lat();
}

}  // namespace robot_interfaces

#endif  // ROBOT_INTERFACES__MSG__DETAIL__EGO_POSE__BUILDER_HPP_
