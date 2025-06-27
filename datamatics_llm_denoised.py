import oci
import mimetypes
import base64
import numpy as np
import os
from PIL import Image, ImageFilter, ImageOps, ImageEnhance

CONFIGFILEPATH='~/.oci/config' # give your config file location here    

def image_to_base642(image_path):
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        raise ValueError("Unable to determine the image MIME type.")
    image = Image.open(image_path)
    gray = ImageOps.grayscale(image)
    enhancer = ImageEnhance.Contrast(gray)
    enhanced = enhancer.enhance(1.5)
    denoised = enhanced.filter(ImageFilter.MedianFilter(size=3))
    directory = os.path.dirname(image_path)
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    
    denoised_filename = f"{name}_denoised{ext}"
    denoised_path = os.path.join(directory, denoised_filename)
    denoised.save(denoised_path)
    print(f"Cleaned image saved to: {denoised_path}")
    with open(denoised_path, "rb") as img_file:
        base64_string = base64.b64encode(img_file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{base64_string}"


def oci_llm_inferencing2(base64img, prompt):
    try:
        config = oci.config.from_file(CONFIGFILEPATH, 'DEFAULT')
        config['region']='us-chicago-1'
        generative_ai_inference_client = oci.generative_ai_inference.GenerativeAiInferenceClient(config)

        chat_response = generative_ai_inference_client.chat(
        chat_details=oci.generative_ai_inference.models.ChatDetails(
        compartment_id="ocid1.tenancy.oc1..aaaaaaaaynhrja6ajmkghmahxk52blanqil7pwxkz44x77g62d2zw4zzqa7q", #your compartment id

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
                                    text=prompt
                                )],
                                )],
                        is_stream=False,
                        seed=592,
                        is_echo=True,
                        top_k=-1,
                        top_p=0.75,
                        temperature=0,
                        frequency_penalty=0,
                        presence_penalty=0
            )),
        )

        # print(chat_response.data)
        return chat_response
    except Exception as e:
        print('llm invocation error')
        print(e)

def run_llm_prompts(filepath, prompt):
    imgbase64=image_to_base642(filepath)
    resp=oci_llm_inferencing2(imgbase64, prompt)
    print(resp.data.chat_response.choices[0].message.content[0].text)

# Example usage:
if __name__ == "__main__":
    imgfilepath='/path/to/imgfile' # your image path

    payloadInvoice1 ="Please return complete JSON only. Extract all fields and the corresponding value co-ordinates in a bounding box in pixels from the text below \
                        using the following the example given to return the JSON. Do not prefix any hypen or star. \
                        Return JSON only. Surround your response in curly brackets {}"
    
    
    payloadInvoice2= """
    \n\nIMPORTANT: All fields are needed. Do not translate.
    
    """
    
    exampleOutput1 = """
{
"First Name ": str,
"First Name_bbox ": Value co-ordiantes in a bounding box,
"Middle Name ": str,
"Middle Name_bbox ": Value co-ordiantes in a bounding box,
"Last Name ": str,
"Last Name_bbox ": Value co-ordiantes in a bounding box,
"Gender ": str,
"Mobile Number 1 (Primary) ": str
"Mobile Number 2 ": str,
"Address Type ": str,
"Permanent Address": str,
"Address_Village": str,
"Address_Tehsil Name": str,
"Address_Post Office": str,
"Address_District": str,
"Address_Sate": str,
"Address_Pin Code": str,
"Pincode": str,
"Residence": str,
"Residence Type": "select the checkboxed item and prefix with 'Semi' if present",
"Loan Amount Requested": str,
"Tenure (Months)": str,
"Loan_Repayment Frequency": "select the checkboxed item",
"DSA Code / Name": str,
"Referral Type": str,
"DST Code / Name": str,
"DSE Code / Name": str,
"Referral Type": str,
"subclass": str,
"Date of Application Receipt": str,
"Source Code": str,
"Co_App_1_First Name ": str,
"Co_App_1_Middle Name ": str,
"Co_App_1_Last Name ": str,
"Co_App_1_Date of Birth ": str,
"Co_App_1_Gender ": str,
"Co_App_1_Relation With Applicant ": str,
"Co_App_1_Mobile Number 1 (Primary) ": str,
"Co_App_1_ Mobile Number 2": str,
"Co_App_1_Permanent Address": str,
"Co_App_1_Permanent Address_Pin Code": str,
"Applicant_Father Name": str,
"Applicant_App_Mothers Maiden Name": str,
"Applicant_Marital Status": str,
"Applicant_Religion": str,
"Applicant_Category": str,
"Applicant_Educational Qualification": str,
"Applicant_Residence Address": str,
"Applicant_Place of Birth": str,
"Co_Applicant_1_Father Name": str,
"Co_Applicant_1_Mothers Maiden Name": str,
"Co_Applicant_1_Marital Status": str,
"Co_Applicant_1_Religion": str,
"Co_Applicant_1_Category": str,
"Co_Applicant_1_Educational Qualification": str,
"Co_Applicant_1_Residence Address Vintage": str,
"Co_Applicant_1_Vintage in city": str,
"Co_Applicant_1_Place of Birth": str,
"Asset_Manufacturer": str,
"Asset_Dealer": str,
"Asset_Model": str,
"Asset_Sub-Model": str,
"Asset_Asset Cost": str,
"Asset_Total Asset Cost": str,
"Asset_Total Asset Cost": str,
"Asset_Loan Amount ": str,
"Loan_Total Loan Amount": str,
"Loan_Loan Amount Sanctioned": str,
"Loan_Tenure (Months)": str,
"Income Declaration_source": "all listed sources",
"Income Declaration_Amount": "all listed Amounts",
"Proposed Security Details_Type": "all listed security type",
"Proposed Security Details_Desc": "all listed security description",
"Proposed Security Details_Value": "all listed security value"

}
  """


  
  
    promptMain=payloadInvoice1 + \
             payloadInvoice2 + \
             exampleOutput1 + \
             "\nResponse:"
  
    # promptMain = "Please extract all field values along with the bounding box."
    output = run_llm_prompts(imgfilepath,promptMain)
    
