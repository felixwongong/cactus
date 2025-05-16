import type { CactusOAICompatibleMessage } from "./chat";
import type { NativeCompletionResult } from "./NativeCactus";

interface Parameter {
  type: string,
  description: string,
  required?: boolean // parameter is optional if not specified
}

interface Tool {
  func: Function,
  description: string,
  parameters: {[key: string]: Parameter},
  required: string[]
}

export class Tools {
  private tools = new Map<string, Tool>();
  
  add(
      func: Function, 
      description: string,
      parameters: {[key: string]: Parameter},
    ) {
      this.tools.set(func.name, { 
        func, 
        description,
        parameters,
        required: Object.entries(parameters)
          .filter(([_, param]) => param.required)
          .map(([key, _]) => key)
      });
      return func;
    }
  
  getSchemas() {
      return Array.from(this.tools.entries()).map(([name, { description, parameters, required }]) => ({
        type: "function",
        function: {
          name,
          description,
          parameters: {
            type: "object",
            properties: parameters,
            required
          }
        }
      }));
    }
  
  async execute(name: string, args: any) {
      const tool = this.tools.get(name);
      if (!tool) throw new Error(`Tool ${name} not found`);
      return await tool.func(...Object.values(args));
  }
}

export function injectToolsIntoMessages(messages: CactusOAICompatibleMessage[], tools: Tools): CactusOAICompatibleMessage[] {
  const newMessages = [...messages];
  const toolsSchemas = tools.getSchemas();
  const promptToolInjection = `You have access to the following functions. Use them if required - 
${JSON.stringify(toolsSchemas, null, 2)}
Only use an available tool if needed. If a tool is chosen, respond ONLY with a JSON object matching the following schema:
\`\`\`json
{
"tool_name": "<name of the tool>",
"tool_input": {
"<parameter_name>": "<parameter_value>",
...
}
}
\`\`\`
Remember, if you are calling a tool, you must respond with the JSON object and the JSON object ONLY!
If no tool is needed, respond normally.
  `;
  
  const systemMessage = newMessages.find(m => m.role === 'system');
  if (!systemMessage) {
      newMessages.unshift({
          role: 'system',
          content: promptToolInjection
      });
  } else {
      systemMessage.content = `${systemMessage.content}\n\n${promptToolInjection}`;
  }
  
  return newMessages;
}

export async function parseAndExecuteTool(result: NativeCompletionResult, tools: Tools): Promise<{toolCalled: boolean, toolName?: string, toolInput?: any, toolOutput?: any}> {
  const match = result.content.match(/```json\s*([\s\S]*?)\s*```/);
  
  if (!match || !match[1]) return {toolCalled: false};
  
  try {
      const jsonContent = JSON.parse(match[1]);
      const { tool_name, tool_input } = jsonContent;
      // console.log('Calling tool:', tool_name, tool_input);
      const toolOutput = await tools.execute(tool_name, tool_input) || true;
      // console.log('Tool called result:', toolOutput);
      
      return {
          toolCalled: true,
          toolName: tool_name,
          toolInput: tool_input,
          toolOutput
      };
  } catch (error) {
      // console.error('Error parsing JSON:', match, error);
      return {toolCalled: false};
  }
}

export function updateMessagesWithToolCall(messages: CactusOAICompatibleMessage[], toolName: string, toolInput: any, toolOutput: any): CactusOAICompatibleMessage[] {
  const newMessages = [...messages];

  newMessages.push({
      role: 'function-call',
      content: JSON.stringify({name: toolName, arguments: toolInput}, null, 2)
  })
  newMessages.push({
      role: 'function-response',
      content: JSON.stringify(toolOutput, null, 2)
  })
  
  return newMessages;
}