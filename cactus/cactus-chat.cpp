#include "cactus.h"

/**
 * @file cactus-chat.cpp
 * @brief Chat formatting functionality for the Cactus LLM interface
 * 
 * This file contains implementations for chat message formatting
 * and template handling.
 */

namespace cactus {

/**
 * @brief Validates if a chat template exists and is valid
 * 
 * @param use_jinja Whether to use Jinja templates
 * @param name Name of the template to validate
 * @return true if template is valid, false otherwise
 */
bool cactus_context::validateModelChatTemplate(bool use_jinja, const char *name) const {
    const char * tmpl = llama_model_chat_template(model, name);
    if (tmpl == nullptr) {
      return false;
    }
    return common_chat_verify_template(tmpl, use_jinja);
}

/**
 * @brief Formats a chat using Jinja templates
 * 
 * @param messages JSON string of chat messages
 * @param chat_template Optional custom chat template
 * @param json_schema JSON schema for validation
 * @param tools JSON string of tools available
 * @param parallel_tool_calls Whether to allow parallel tool calls
 * @param tool_choice Tool choice preference
 * @return Formatted chat parameters
 */
common_chat_params cactus_context::getFormattedChatWithJinja(
  const std::string &messages,
  const std::string &chat_template,
  const std::string &json_schema,
  const std::string &tools,
  const bool &parallel_tool_calls,
  const std::string &tool_choice
) const {
    common_chat_templates_inputs inputs;
    inputs.use_jinja = true;
    inputs.messages = common_chat_msgs_parse_oaicompat(json::parse(messages));
    auto useTools = !tools.empty();
    if (useTools) {
        inputs.tools = common_chat_tools_parse_oaicompat(json::parse(tools));
    }
    inputs.parallel_tool_calls = parallel_tool_calls;
    if (!tool_choice.empty()) {
        inputs.tool_choice = common_chat_tool_choice_parse_oaicompat(tool_choice);
    }
    if (!json_schema.empty()) {
        inputs.json_schema = json::parse(json_schema);
    }
    inputs.extract_reasoning = params.reasoning_format != COMMON_REASONING_FORMAT_NONE;

    // If chat_template is provided, create new one and use it (probably slow)
    if (!chat_template.empty()) {
        auto tmps = common_chat_templates_init(model, chat_template);
        return common_chat_templates_apply(tmps.get(), inputs);
    } else {
        return common_chat_templates_apply(templates.get(), inputs);
    }
}

/**
 * @brief Formats a chat using standard templates
 * 
 * @param messages JSON string of chat messages
 * @param chat_template Optional custom chat template
 * @return Formatted prompt string
 */
std::string cactus_context::getFormattedChat(
  const std::string &messages,
  const std::string &chat_template
) const {
    common_chat_templates_inputs inputs;
    inputs.messages = common_chat_msgs_parse_oaicompat(json::parse(messages));
    inputs.use_jinja = false;

    // If chat_template is provided, create new one and use it (probably slow)
    if (!chat_template.empty()) {
        auto tmps = common_chat_templates_init(model, chat_template);
        return common_chat_templates_apply(tmps.get(), inputs).prompt;
    } else {
        return common_chat_templates_apply(templates.get(), inputs).prompt;
    }
}

} // namespace cactus 