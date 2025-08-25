import datetime
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langgraph.types import RetryPolicy
from pydantic import SecretStr
from data.State import State
from tool.screen_content import *

os.environ["LANGCHAIN_TRACING_V2"] = config.LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_ENDPOINT"] = config.LANGCHAIN_ENDPOINT
os.environ["LANGCHAIN_API_KEY"] = config.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = config.LANGCHAIN_PROJECT

model = ChatOpenAI(
    openai_api_base=config.LLM_BASE_URL,
    openai_api_key=SecretStr(config.LLM_API_KEY),
    model_name=config.LLM_MODEL,
    request_timeout=config.LLM_REQUEST_TIMEOUT,
    max_retries=config.LLM_MAX_RETRIES,
    max_tokens=config.LLM_MAX_TOKEN,
)


def tsk_setting(state: State):
    # Task-related settings
    try:
        print(f"🔧 tsk_setting: Processing task: '{state.get('tsk', 'NO TASK')}'")
        
        if not state.get("tsk"):
            print(f"❌ Error: No task provided in state")
            state["errors"] = state.get("errors", [])
            state["errors"].append({"step": "tsk_setting", "error": "No task provided"})
            return state
            
        message = [
            SystemMessage("Please reply with only the application name"),
            HumanMessage(
                f"The task goal is: {state['tsk']}, please infer the related application name. (The application name should not contain spaces) and reply with only one"
            ),
        ]
        llm_response = model.invoke(message)
        app_name = llm_response.content
        state["app_name"] = app_name
        print(f"🔧 tsk_setting: Inferred app name: '{app_name}'")
        
        state["context"] = [
            HumanMessage(
                f"The task goal is: {state['tsk']}, the inferred application name is: {app_name}"
            )
        ]

        # Validate device parameter before calling get_device_size
        device = state.get("device", "emulator-5554")
        if not isinstance(device, str):
            print(f"❌ Error in tsk_setting: Device parameter is not a string: {type(device)} - {device}")
            device = "emulator-5554"
            print(f"🔄 Using fallback device: {device}")
            state["device"] = device  # Update state with corrected device
        
        print(f"🔧 tsk_setting: Getting device size for: '{device}'")
        device_info = get_device_size.invoke(device)
        state["device_info"] = device_info
        print(f"🔧 tsk_setting: Device info: {device_info}")

        # Prepare additional information to pass to the callback function
        callback_info = {
            "app_name": state["app_name"],
            "device_info": state["device_info"],
            "task": state["tsk"],
        }

        # Call the callback function (if any)
        if state.get("callback"):
            # Pass both the current node name and additional information to the callback function
            state["callback"](state, node_name="tsk_setting", info=callback_info)

        print(f"🔧 tsk_setting: Completed successfully")
        return state
        
    except Exception as e:
        print(f"❌ Error in tsk_setting: {str(e)}")
        state["errors"] = state.get("errors", [])
        state["errors"].append({"step": "tsk_setting", "error": str(e)})
        import traceback
        print(f"❌ Traceback: {traceback.format_exc()}")
        return state


def page_understand(state: State):
    """
    Understand the current page
    """
    try:
        print(f"📸 page_understand: Starting for step {state.get('step', 0)}")
        
        # Validate device parameter before calling take_screenshot
        device = state.get("device", "emulator-5554")
        if not isinstance(device, str):
            print(f"❌ Error in page_understand: Device parameter is not a string: {type(device)} - {device}")
            device = "emulator-5554"
            print(f"🔄 Using fallback device: {device}")
        
        app_name = state.get("app_name", "unknown_app")
        print(f"📸 page_understand: Taking screenshot for device='{device}', app='{app_name}', step={state.get('step', 0)}")
        
        screen_img = take_screenshot.invoke(
            {
                "device": device,
                "app_name": app_name,
                "step": state["step"],
            }
        )
        
        # Check if screenshot was successful
        if "failed" in screen_img.lower() or "error" in screen_img.lower():
            print(f"❌ Screenshot failed: {screen_img}")
            state["errors"] = state.get("errors", [])
            state["errors"].append({"step": "page_understand", "error": f"Screenshot failed: {screen_img}"})
            return state
            
        print(f"📸 page_understand: Screenshot saved to: {screen_img}")
        
        print(f"🔍 page_understand: Parsing screen elements...")
        screen_result = screen_element.invoke(
            {
                "image_path": screen_img,
            }
        )
        
        # Check if screen parsing was successful
        if "error" in screen_result:
            print(f"❌ Screen parsing failed: {screen_result['error']}")
            state["errors"] = state.get("errors", [])
            state["errors"].append({"step": "page_understand", "error": f"Screen parsing failed: {screen_result['error']}"})
            return state
            
        print(f"🔍 page_understand: Parsing successful, JSON saved to: {screen_result.get('parsed_content_json_path', 'NOT FOUND')}")
        
        state["current_page_screenshot"] = screen_img
        state["current_page_json"] = screen_result["parsed_content_json_path"]
        
        # Update page_history with the screenshot path
        if not isinstance(state["page_history"], list):
            state["page_history"] = []
        state["page_history"].append(screen_img)
        
        # Call the callback function (if any)
        if state.get("callback"):
            state["callback"](state, node_name="page_understand")

        # Add tool result to state
        if not isinstance(state["tool_results"], list):
            state["tool_results"] = []

        state["tool_results"].append(
            {"tool_name": "screen_element", "result": screen_result}
        )

        print(f"📸 page_understand: Completed successfully")
        return state
        
    except Exception as e:
        print(f"❌ Error in page_understand: {str(e)}")
        state["errors"] = state.get("errors", [])
        state["errors"].append({"step": "page_understand", "error": str(e)})
        import traceback
        print(f"❌ Traceback: {traceback.format_exc()}")
        return state


def perform_action(state: State):
    """
    Perform actions based on the current state
    Specifically, this node does two things:
    1. Use LLM to understand the interface and generate recommended actions (based on the current page screenshot, parsed JSON, and user intent)
    2. Execute the recommended action (by calling the relevant tools through the React agent)

    In this step:
    - Need to get the annotated screenshot and parsed JSON result from state
    - Pass these data along with the user's intent to LLM, let React agent analyze and decide
    - Execute the corresponding tool operation
    - Update state's step count, history information, etc.
    """

    # Create action_agent, used for decision making and executing operations on the page
    action_agent = create_react_agent(model, [screen_action])

    # Get the annotated screenshot path and parsed JSON data from state
    labeled_image_path = state.get("current_page_screenshot")
    json_labeled_path = state.get("current_page_json")
    user_intent = state.get("tsk", "No specific task")
    device = state.get("device", "emulator-5554")
    device_size = state.get("device_info", {})
    
    # Validate that device is a string, not a function or other object
    if not isinstance(device, str):
        print(f"❌ Error: Device parameter is not a string: {type(device)} - {device}")
        # Fall back to default device
        device = "emulator-5554"
        print(f"🔄 Falling back to default device: {device}")

    # Read screenshot file and encode to base64
    with open(labeled_image_path, "rb") as f:
        image_content = f.read()
    image_data = base64.b64encode(image_content).decode("utf-8")

    # Read parsed JSON file content
    with open(json_labeled_path, "r", encoding="utf-8") as f:
        page_json = f.read()

    # Build message list to pass to LLM, including user intent, page parsed result, and screenshot information
    messages = [
        SystemMessage(
            content=f"Below is the current page information and user intent, please analyze and recommend a reasonable next step based on this. Please only complete one step."
            f"All tool calls must include device to specify the operation device."
        ),
        HumanMessage(
            content=f"The current device is: {device}, the screen size of the device is {device_size}."
            f"The current task intent is: {user_intent}"
        ),
        HumanMessage(
            content="Below is the parsed JSON data of the current page (the bbox is relative, please convert it to actual operation position based on screen size): \n"
            + page_json
        ),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Below is the base64 data of the annotated page screenshot:",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ],
        ),
    ]

    # Add these messages to state's context for maintaining dialog continuity
    state["context"].extend(messages)

    # Call action_agent for decision and execute operation
    action_result = action_agent.invoke({"messages": state["context"][-4:]})

    # The last message of final_message as the final decision output
    final_messages = action_result.get("messages", [])
    if final_messages:
        # Add final_message to context for continuity
        state["context"].append(final_messages[-1])
    else:
        # If no return message, it indicates an error
        state["context"].append(
            SystemMessage(content="No action decided due to an error.")
        )
        state["errors"].append(
            {"step": state["step"], "error": "No messages returned by action_agent"}
        )
        return state

    # Extract recommended action and execution status from final_message
    recommended_action = final_messages[-1].content.strip()
    state["recommend_action"] = recommended_action

    # Parse all tool_messages to get tool execution results
    tool_messages = [msg for msg in final_messages if msg.type == "tool"]
    tool_output = {}
    for tool_message in tool_messages:
        tool_output.update(json.loads(tool_message.content))

    if tool_output:
        # Ensure tool_results is a list
        if not isinstance(state["tool_results"], list):
            state["tool_results"] = []
        # Add tool name for front-end recognition
        tool_output["tool_name"] = "screen_action"  # Or other corresponding tool name
        state["tool_results"].append(tool_output)

    # Update step counter BEFORE adding to history
    state["step"] += 1

    # Add this operation record to history step record for future query
    step_record = {
        "step": state["step"] - 1,  # Use the step that was just executed
        "recommended_action": recommended_action,
        "tool_result": tool_output,
        "source_page": state["current_page_screenshot"],
        "source_json": state["current_page_json"],
        "timestamp": datetime.datetime.now().isoformat(),
    }
    state["history_steps"].append(step_record)

    # Call callback
    if state.get("callback"):
        state["callback"](state, node_name="perform_action")

    return state


def tsk_completed(state: State):
    """
    When the number of execution steps exceeds three, use LLM and the current history of three screenshots to determine if the task is complete.
    It consists of two steps:
    1. Use the user's task itself to reflect on the completion criteria (generate a description of the completion criteria using LLM)
    2. Use the three screenshots and the completion criteria generated in the first step to ask LLM to judge whether the task is completed.
    """

    # If step is less than 3, no judgment is made, directly return current completion status
    if state["step"] < 2:
        return state["completed"]

    # Get user task description
    user_task = state.get("tsk", "No task description")

    # First step: let LLM reflect on user task, generate task completion judgment criteria
    reflection_messages = [
        SystemMessage(
            content="You are a supportive intelligent assistant, helping to analyze task completion criteria."
        ),
        HumanMessage(
            content=f"The user's task is: {user_task}\nPlease describe clear and checkable task completion criteria. For example: 'When a certain element or status appears on the page, it indicates that the task is completed.'"
        ),
    ]

    # Call LLM to generate completion criteria
    reflection_response = model.invoke(reflection_messages)
    completion_criteria = reflection_response.content.strip()

    # Add generated completion criteria to context
    state["context"].append(
        SystemMessage(
            content=f"Generated task completion judgment criteria: {completion_criteria}"
        )
    )

    # Second step: use the latest three page screenshots and completion criteria to ask LLM to judge
    # Ensure enough screenshot history
    if len(state["page_history"]) < 3:
        # If not enough three screenshots, use as many existing screenshots as possible.
        # But logically >=3 steps should at least have 3 times page_understand screenshots, here for safety make downgrade processing.
        recent_images = state["page_history"]
    else:
        recent_images = state["page_history"][-3:]

    # Convert three screenshots to base64 and package as LLM messages
    image_messages = []
    for idx, img_path in enumerate(recent_images, start=1):
        if os.path.exists(img_path):
            with open(img_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")
            image_messages.append(
                HumanMessage(
                    content=[
                        {
                            "type": "text",
                            "text": f"Below is the base64 data of the {idx}th screenshot:",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_data}"},
                        },
                    ]
                )
            )
        else:
            # If path does not exist, send a descriptive text
            image_messages.append(
                HumanMessage(
                    content=f"Cannot find the {idx}th screenshot path: {img_path}"
                )
            )

    # Build final judgment dialog information
    judgement_messages = [
        SystemMessage(
            content="You are a page judgment assistant, you will judge whether the task is completed based on the given completion criteria and current page screenshot."
        ),
        HumanMessage(
            content=f"Completion criteria: {completion_criteria}\n"
            f"Please judge whether the task is completed based on the following three page screenshots. Please note that if all three screenshots are complete, it indicates task failure, please directly reply yes or complete to end the program."
        ),
    ] + image_messages

    # Call LLM for final judgment
    judgement_response = model.invoke(judgement_messages)
    judgement_answer = judgement_response.content.strip()

    # Update state["completed"] based on LLM's answer
    # Assuming LLM answers "yes" or "no", of course, it can be conditionally adapted based on actual model output
    if "yes" in judgement_answer or "complete" in judgement_answer.lower():
        state["completed"] = True
        # Take final screenshot and parse to capture the completion state
        screen_img = take_screenshot.invoke(
            {
                "device": state["device"],
                "app_name": state["app_name"],
                "step": state["step"],
            }
        )
        screen_result = screen_element.invoke(
            {
                "image_path": screen_img,
            }
        )
        state["current_page_screenshot"] = screen_img
        state["current_page_json"] = screen_result["parsed_content_json_path"]
        
        # Update page_history with final screenshot
        if not isinstance(state["page_history"], list):
            state["page_history"] = []
        state["page_history"].append(screen_img)
    else:
        state["completed"] = False

    # Add final judgment to context
    state["context"].append(
        SystemMessage(
            content=f"LLM's answer on whether the task is completed: {judgement_answer}"
        )
    )
    state["context"].append(
        SystemMessage(content=f"Final task completion status: {state['completed']}")
    )

    if state["step"] > 5:  # Debug use - force completion after 5 steps
        screen_img = take_screenshot.invoke(
            {
                "device": state["device"],
                "app_name": state["app_name"],
                "step": state["step"],
            }
        )
        screen_result = screen_element.invoke(
            {
                "image_path": screen_img,
            }
        )
        state["current_page_screenshot"] = screen_img
        state["current_page_json"] = screen_result["parsed_content_json_path"]
        
        # Update page_history with final screenshot for debug completion
        if not isinstance(state["page_history"], list):
            state["page_history"] = []
        state["page_history"].append(screen_img)
        
        state["completed"] = True
        return True
    return state["completed"]


# User interaction interface
def run_task(initial_state: State, progress_callback=None):
    # Build StateGraph
    graph_builder = StateGraph(State)
    # Define nodes in the graph
    graph_builder.add_node(
        "tsk_setting", tsk_setting, retry=RetryPolicy(max_attempts=5)
    )
    graph_builder.add_node(
        "page_understand", page_understand, retry=RetryPolicy(max_attempts=5)
    )
    graph_builder.add_node("perform_action", perform_action)

    # Define edges in the graph
    graph_builder.add_edge(START, "tsk_setting")
    graph_builder.add_edge("tsk_setting", "page_understand")
    graph_builder.add_conditional_edges(
        "page_understand", tsk_completed, {True: END, False: "perform_action"}
    )
    graph_builder.add_edge("perform_action", "page_understand")

    # Compile graph
    graph = graph_builder.compile()

    # Visualize graph
    # graph.get_graph().draw_mermaid_png(output_file_path="graph_vis.png")

    # Put callback into state
    if progress_callback is not None:
        initial_state["callback"] = progress_callback

    result = graph.invoke(initial_state)
    return result
