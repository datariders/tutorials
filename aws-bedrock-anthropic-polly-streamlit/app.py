import boto3
import json 
import os   
from botocore.exceptions import ClientError
import streamlit as st
            
            
# Set up the Amazon Bedrock client
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1',  # Replace with your AWS region
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY")
)               
                
# Set up the Amazon Polly client
polly_client = boto3.client(
    service_name='polly',
    region_name='us-east-1',  # Replace with your AWS region
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY")
)                   
                
# Specify the Claude 3.5 Sonnet model ID
model_id = 'anthropic.claude-3-5-sonnet-20240620-v1:0'
            
def generate_speech(text, voice_id='Joanna'):
    try:
        # Call Polly to synthesize the text into speech
        response = polly_client.synthesize_speech(
            Text=text,
            OutputFormat='mp3',  # You can also use 'ogg_vorbis' or 'pcm'
            VoiceId=voice_id
        )
        
        # Save the speech to an MP3 file
        audio_file = "output_speech.mp3"
        with open(audio_file, 'wb') as file:
            file.write(response['AudioStream'].read())

        return audio_file
    except ClientError as e:
        st.error(f"Polly error: {e}")
        return None


def main():
    st.title("MathNEXT: Personal Math Tutor")

    user_query = st.text_input("Enter your query:")
    if user_query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        exit()

    if user_query:
        try:
            # Prepare the request body
            request_body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 300,
                "temperature": 0.7,
                "messages": [
                    {"role": "user", "content": user_query}
                ]
            })

            # Invoke the Bedrock model
            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                body=request_body
            )

            # Parse the response
            response_body = json.loads(response['body'].read())
            output = response_body['content'][0]['text']

            # Display Bedrock model response
            st.write("MathNEXT response")
            st.write(output)

            # Generate speech from the Bedrock response
            audio_file = generate_speech(output)

            if audio_file:
                # Provide a link to download the speech
                with open(audio_file, 'rb') as file:
                    st.audio(file.read(), format="audio/mp3")

                st.download_button(
                    label="Download Speech",
                    data=open(audio_file, 'rb'),
                    file_name=audio_file,
                    mime='audio/mp3'
                )

        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            st.error(f"Error: {error_code} - {error_message}")
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")

        # Print debugging information
        print("\nDebugging Information:")
        print(f"AWS Region: {bedrock_runtime.meta.region_name}")
        print(f"Model used: {model_id}")
        print("Please ensure you have the correct permissions and that this model is available in your region.")


if _name_ == "_main_":
    main()
