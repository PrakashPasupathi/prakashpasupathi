import oci
import mimetypes
import base64
CONFIGFILEPATH='' # give your config file location here

def image_to_base642(image_path):
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        raise ValueError("Unable to determine the image MIME type.")
    with open(image_path, "rb") as img_file:
        base64_string = base64.b64encode(img_file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{base64_string}"


    def image_to_base642(image_path):
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        raise ValueError("Unable to determine the image MIME type.")
    with open(image_path, "rb") as img_file:
        base64_string = base64.b64encode(img_file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{base64_string}"

def oci_llm_inferencing2(base64img):
    try:
        config = oci.config.from_file(CONFIGFILEPATH, 'DEFAULT')
        config['region']='us-chicago-1'
        generative_ai_inference_client = oci.generative_ai_inference.GenerativeAiInferenceClient(config)
        
        chat_response = generative_ai_inference_client.chat(
        chat_details=oci.generative_ai_inference.models.ChatDetails(
        compartment_id="", #your compartment id
        serving_mode=oci.generative_ai_inference.models.OnDemandServingMode(
            serving_type="ON_DEMAND",
            model_id="meta.llama-4-scout-17b-16e-instruct"), # model id
        chat_request=oci.generative_ai_inference.models.GenericChatRequest(
            api_format="GENERIC",
            messages=[
                oci.generative_ai_inference.models.UserMessage(
                    role="USER",
                    content=[
                        oci.generative_ai_inference.models.ImageContent(
                                    type="IMAGE",
                                    image_url=oci.generative_ai_inference.models.ImageUrl(
                                        url=base64img,
                                        detail="AUTO")),
                            oci.generative_ai_inference.models.TextContent(
                                type="TEXT",
                                text="""Please extract all field values along with their bounding box."""
                            )],
                            )],
                    is_stream=False,
                    seed=592,
                    is_echo=True,
                    top_k=-1,
                    top_p=0.75,
                    temperature=0,
                    frequency_penalty=0,
                    presence_penalty=0,

                    max_tokens=1000
        )),
            )

        # print(chat_response.data)
        return chat_response
    except Exception as e:
        print('llm invocation error')
        print(e)


imgfilepath='' # your image path
imgbase64=image_to_base642(imgfilepath)
resp=oci_llm_inferencing2(imgbase64)

print(resp.data.chat_response.choices[0].message.content[0].text)