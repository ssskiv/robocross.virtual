// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from robot_interfaces:srv/PoseService.idl
// generated code does not contain a copyright notice

#ifndef ROBOT_INTERFACES__SRV__DETAIL__POSE_SERVICE__TRAITS_HPP_
#define ROBOT_INTERFACES__SRV__DETAIL__POSE_SERVICE__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "robot_interfaces/srv/detail/pose_service__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'request'
#include "std_msgs/msg/detail/string__traits.hpp"

namespace robot_interfaces
{

namespace srv
{

inline void to_flow_style_yaml(
  const PoseService_Request & msg,
  std::ostream & out)
{
  out << "{";
  // member: request
  {
    out << "request: ";
    to_flow_style_yaml(msg.request, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const PoseService_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: request
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "request:\n";
    to_block_style_yaml(msg.request, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const PoseService_Request & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace robot_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use robot_interfaces::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const robot_interfaces::srv::PoseService_Request & msg,
  std::ostream & out, size_t indentation = 0)
{
  robot_interfaces::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use robot_interfaces::srv::to_yaml() instead")]]
inline std::string to_yaml(const robot_interfaces::srv::PoseService_Request & msg)
{
  return robot_interfaces::srv::to_yaml(msg);
}

template<>
inline const char * data_type<robot_interfaces::srv::PoseService_Request>()
{
  return "robot_interfaces::srv::PoseService_Request";
}

template<>
inline const char * name<robot_interfaces::srv::PoseService_Request>()
{
  return "robot_interfaces/srv/PoseService_Request";
}

template<>
struct has_fixed_size<robot_interfaces::srv::PoseService_Request>
  : std::integral_constant<bool, has_fixed_size<std_msgs::msg::String>::value> {};

template<>
struct has_bounded_size<robot_interfaces::srv::PoseService_Request>
  : std::integral_constant<bool, has_bounded_size<std_msgs::msg::String>::value> {};

template<>
struct is_message<robot_interfaces::srv::PoseService_Request>
  : std::true_type {};

}  // namespace rosidl_generator_traits

// Include directives for member types
// Member 'response'
#include "robot_interfaces/msg/detail/ego_pose__traits.hpp"

namespace robot_interfaces
{

namespace srv
{

inline void to_flow_style_yaml(
  const PoseService_Response & msg,
  std::ostream & out)
{
  out << "{";
  // member: response
  {
    out << "response: ";
    to_flow_style_yaml(msg.response, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const PoseService_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: response
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "response:\n";
    to_block_style_yaml(msg.response, out, indentation + 2);
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const PoseService_Response & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace srv

}  // namespace robot_interfaces

namespace rosidl_generator_traits
{

[[deprecated("use robot_interfaces::srv::to_block_style_yaml() instead")]]
inline void to_yaml(
  const robot_interfaces::srv::PoseService_Response & msg,
  std::ostream & out, size_t indentation = 0)
{
  robot_interfaces::srv::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use robot_interfaces::srv::to_yaml() instead")]]
inline std::string to_yaml(const robot_interfaces::srv::PoseService_Response & msg)
{
  return robot_interfaces::srv::to_yaml(msg);
}

template<>
inline const char * data_type<robot_interfaces::srv::PoseService_Response>()
{
  return "robot_interfaces::srv::PoseService_Response";
}

template<>
inline const char * name<robot_interfaces::srv::PoseService_Response>()
{
  return "robot_interfaces/srv/PoseService_Response";
}

template<>
struct has_fixed_size<robot_interfaces::srv::PoseService_Response>
  : std::integral_constant<bool, has_fixed_size<robot_interfaces::msg::EgoPose>::value> {};

template<>
struct has_bounded_size<robot_interfaces::srv::PoseService_Response>
  : std::integral_constant<bool, has_bounded_size<robot_interfaces::msg::EgoPose>::value> {};

template<>
struct is_message<robot_interfaces::srv::PoseService_Response>
  : std::true_type {};

}  // namespace rosidl_generator_traits

namespace rosidl_generator_traits
{

template<>
inline const char * data_type<robot_interfaces::srv::PoseService>()
{
  return "robot_interfaces::srv::PoseService";
}

template<>
inline const char * name<robot_interfaces::srv::PoseService>()
{
  return "robot_interfaces/srv/PoseService";
}

template<>
struct has_fixed_size<robot_interfaces::srv::PoseService>
  : std::integral_constant<
    bool,
    has_fixed_size<robot_interfaces::srv::PoseService_Request>::value &&
    has_fixed_size<robot_interfaces::srv::PoseService_Response>::value
  >
{
};

template<>
struct has_bounded_size<robot_interfaces::srv::PoseService>
  : std::integral_constant<
    bool,
    has_bounded_size<robot_interfaces::srv::PoseService_Request>::value &&
    has_bounded_size<robot_interfaces::srv::PoseService_Response>::value
  >
{
};

template<>
struct is_service<robot_interfaces::srv::PoseService>
  : std::true_type
{
};

template<>
struct is_service_request<robot_interfaces::srv::PoseService_Request>
  : std::true_type
{
};

template<>
struct is_service_response<robot_interfaces::srv::PoseService_Response>
  : std::true_type
{
};

}  // namespace rosidl_generator_traits

#endif  // ROBOT_INTERFACES__SRV__DETAIL__POSE_SERVICE__TRAITS_HPP_
