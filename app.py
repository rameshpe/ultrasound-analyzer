import base64
import html
import io
import uuid
from pathlib import Path
from typing import List

import requests
import streamlit as st
from fpdf import FPDF
from PIL import Image


REPORT_HEADINGS = [
    "Image Type",
    "Major findings",
    "Root Cause",
    "Next Steps",
    "Conclusion",
    "Disclaimer",
]


st.set_page_config(
    page_title="Ultrasound Analyzer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: flex-start;
    }
    .chat-message.user {
        background-color: #e3f2fd;
        border-left: 4px solid #2196F3;
    }
    .chat-message.assistant {
        background-color: #f5f5f5;
        border-left: 4px solid #4CAF50;
    }
    .chat-message.error {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
    </style>
""",
    unsafe_allow_html=True,
)


class OVMSClient:
    def __init__(self, endpoint: str, model_name: str, timeout_seconds: int = 180) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _image_to_data_url(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    def infer(self, prompt: str, images: List[Image.Image], max_tokens: int = 1024) -> str:
        if not images:
            raise ValueError("At least one image is required for EchoVLM_V2 inference")

        content = [{"type": "text", "text": prompt}]
        for img in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._image_to_data_url(img.convert("RGB"))},
                }
            )

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are an expert radiologist in interpreting ultrasound images and creating detailed medical reports."},
                {"role": "user", "content": content},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        }

        response = requests.post(self.endpoint, json=payload, timeout=self.timeout_seconds)
        response.raise_for_status()
        data = response.json()

        if "choices" not in data or not data["choices"]:
            raise RuntimeError(f"Unexpected OVMS response: {data}")

        return data["choices"][0]["message"]["content"].strip()


@st.cache_resource
def get_ovms_client(endpoint: str, model_name: str, timeout_seconds: int) -> OVMSClient:
    return OVMSClient(endpoint=endpoint, model_name=model_name, timeout_seconds=timeout_seconds)


def generate_report(client: OVMSClient, images: List[Image.Image]) -> str:
    prompt = f"""
Describe any abnormalities or significant findings in the provided ultrasound image(s).
If multiple images are provided, synthesize findings across all images.
Create a detailed medical report without patient name, date, or personal details.
The report must contain only the following sections:
**Major findings**: 
<Describe any abnormalities, suspicious regions, or notable features observed in the ultrasound image(s).>
**Root Cause**: 
<Explain the underlying cause of the observed abnormalities or findings.>
**Next Steps**: 
<Recommend further diagnostic tests, treatments, or follow-up actions.>
**Conclusion**: 
<Summarize the overall assessment and key takeaways from the ultrasound images.>
**Disclaimer**: 
<Include any necessary disclaimers regarding the limitations of the analysis or report.>
"""
    return client.infer(prompt=prompt.strip(), images=images, max_tokens=1024)


def answer_user_question(client: OVMSClient, images: List[Image.Image], question: str) -> str:
    scoped_prompt = (
        "Answer the user query using only the provided medical image(s). "
        "If details are uncertain, explicitly say so.\n\n"
        f"User query: {question}"
    )
    return client.infer(prompt=scoped_prompt, images=images, max_tokens=768)


def _normalize_report_lines(report: str) -> List[str]:
    return [line.strip().replace("**", "") for line in report.splitlines()]


def format_report_for_chat(report: str) -> str:
    formatted_lines: List[str] = []
    for line in _normalize_report_lines(report):
        if not line:
            formatted_lines.append("")
            continue

        matched_heading = None
        heading_content = ""
        for heading in REPORT_HEADINGS:
            heading_with_colon = f"{heading}:"
            if line.lower().startswith(heading_with_colon.lower()):
                matched_heading = heading
                heading_content = line[len(heading_with_colon):].strip()
                break
            if line.lower() == heading.lower():
                matched_heading = heading
                break

        if matched_heading:
            if heading_content:
                formatted_lines.append(f"**{matched_heading}:** {heading_content}")
            else:
                formatted_lines.append(f"**{matched_heading}:**")
        else:
            formatted_lines.append(line)

    return "\n".join(formatted_lines).strip()


def _is_heading_line(line: str) -> bool:
    for heading in REPORT_HEADINGS:
        if line.lower() == heading.lower() or line.lower().startswith(f"{heading.lower()}:"):
            return True
    return False


def _safe_chat_content(content: str) -> str:
    return html.escape(content).replace("\n", "<br>")


def create_pdf_report(report: str, images: List[Image.Image]) -> bytes:
    pdf = FPDF()
    pdf.set_font("helvetica", size=12)

    for image in images:
        pdf.add_page()
        temp_path = Path(f"{uuid.uuid4()}.png")
        image.save(temp_path)
        pdf.image(str(temp_path), x=20, y=20, w=170)
        temp_path.unlink(missing_ok=True)

    pdf.add_page()
    for line in _normalize_report_lines(report):
        if not line:
            pdf.ln(4)
            continue

        if _is_heading_line(line):
            pdf.set_font("helvetica", style="B", size=12)
            pdf.multi_cell(0, 7, line)
        else:
            pdf.set_font("helvetica", style="", size=12)
            pdf.multi_cell(0, 7, line)

    pdf_output = pdf.output(dest="S")
    if isinstance(pdf_output, str):
        return pdf_output.encode("latin-1")
    return bytes(pdf_output)


def main() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    if "last_pdf_bytes" not in st.session_state:
        st.session_state.last_pdf_bytes = None

    if "last_pdf_filename" not in st.session_state:
        st.session_state.last_pdf_filename = "echo_vlm_report.pdf"

    with st.sidebar:
        st.title("⚙️ Configuration")
        st.markdown("---")

        st.subheader("OVMS Endpoint")
        ovms_endpoint = st.text_input(
            "Chat Completions URL",
            value="http://localhost:8000/v3/chat/completions",
            help="OpenVINO Model Server chat-completions endpoint URL",
        )
        model_name = st.text_input(
            "Model Name",
            value="EchoVLM_V2",
            help="Model name exposed by OVMS",
        )
        timeout_seconds = st.number_input("Request Timeout (sec)", min_value=30, max_value=900, value=180)

        st.markdown("---")
        st.subheader("Analysis Options")
        include_pdf = st.checkbox("📄 Generate PDF Report", value=True)
        #auto_analyze = st.checkbox("🤖 Auto-analyze image type and generate report", value=True)

        st.markdown("---")
        st.subheader("Session")
        if st.button("🔄 New Chat Session"):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.last_pdf_bytes = None
            st.rerun()

        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.last_pdf_bytes = None
            st.rerun()

        st.markdown("---")
        st.subheader("📖 Instructions")
        st.markdown(
            """
1. Upload one or multiple medical images
2. Ask a question in chat
3. App sends all selected images to EchoVLM_V2 on OVMS
4. Optionally auto-generates structured medical report
"""
        )

    st.title("🏥 EchoVLM_V2 Medical Image Analysis")
    st.markdown("Inference is executed remotely on OpenVINO Model Server.")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📸 Upload Medical Image(s)")
        uploaded_files = st.file_uploader(
            "Choose one or more medical images",
            type=["jpg", "jpeg", "png", "bmp", "gif"],
            accept_multiple_files=True,
            key="image_uploader",
        )

        selected_images: List[Image.Image] = []
        if uploaded_files:
            for uploaded in uploaded_files:
                selected_images.append(Image.open(uploaded).convert("RGB"))

            st.session_state.current_images = selected_images
            st.session_state.current_image_names = [f.name for f in uploaded_files]

            preview_tabs = st.tabs([f"Image {i + 1}" for i in range(len(selected_images))])
            for index, tab in enumerate(preview_tabs):
                with tab:
                    st.image(selected_images[index], use_container_width=True, caption=st.session_state.current_image_names[index])
        else:
            st.session_state.current_images = []
            st.session_state.current_image_names = []

    with col2:
        st.subheader("📊 Image Stats")
        if st.session_state.current_images:
            st.success(f"✅ {len(st.session_state.current_images)} image(s) loaded")
            first = st.session_state.current_images[0]
            st.info(f"First image: {first.width}×{first.height}")
        else:
            st.warning("⚠️ No images selected")

    st.markdown("---")
    st.subheader("💬 Chat Interface")

    if st.session_state.last_pdf_bytes:
        st.download_button(
            label="📄 Download PDF Report",
            data=st.session_state.last_pdf_bytes,
            file_name=st.session_state.last_pdf_filename,
            mime="application/pdf",
            use_container_width=True,
            key="download_pdf_report",
        )

    for message in st.session_state.messages:
        if message["role"] == "user":
            safe_content = _safe_chat_content(message["content"])
            st.markdown(
                f"""
                <div class="chat-message user">
                    <div style="flex: 1;">
                        <b>👤 You:</b><br>{safe_content}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            safe_content = _safe_chat_content(message["content"])
            st.markdown(
                f"""
                <div class="chat-message assistant">
                    <div style="flex: 1;">
                        <b>🏥 Assistant:</b><br>{safe_content}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    col_input, col_send = st.columns([6, 1])
    with col_input:
        user_input = st.text_input(
            "Ask a question about the selected image(s):",
            placeholder="e.g., 'Across these ultrasound frames, is there any suspicious region?'",
            key="user_input",
        )
    with col_send:
        st.write("")
        send_button = st.button("📤 Send", use_container_width=True, key="send_button")

    if send_button and user_input:
        if not st.session_state.current_images:
            st.error("❌ Please upload at least one image first!")
        else:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.spinner("🔄 Running EchoVLM inference on OVMS..."):
                try:
                    client = get_ovms_client(ovms_endpoint, model_name, timeout_seconds)
                    images = st.session_state.current_images

                    if include_pdf:
                        report = generate_report(client, images)
                        formatted_report = format_report_for_chat(report)
                        response = f"""
**Medical Report:**
{formatted_report}
"""

                    
                        pdf_bytes = create_pdf_report(report, images)
                        st.session_state.last_pdf_bytes = pdf_bytes
                        st.session_state.last_pdf_filename = "echo_vlm_report.pdf"
                        
                    else:
                        response = answer_user_question(client, images, user_input)
                        st.session_state.last_pdf_bytes = None

                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()
                except Exception as exc:
                    error_message = f"❌ Error during OVMS inference: {exc}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center; color: gray; font-size: 12px;">
        <p>Ultrasound Analyzer using OpenVINO Model Server</p>
        <p>⚠️ For demonstration purposes only. Always consult healthcare professionals for diagnosis.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
