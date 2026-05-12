"""
GenGuard-AI Streamlit Application
Dashboard for testing and monitoring the framework.
"""

import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="GenGuard-AI Dashboard",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ GenGuard-AI: Privacy-Preserving AI Framework")
st.markdown("---")

if "pipeline" not in st.session_state:
    from genGuard.pipeline import GenGuardPipeline, PipelineConfig

    config = PipelineConfig(
        enable_dp_training=True,
        enable_watermarking=True,
        enable_pii_detection=True,
        privacy_budget=8.0,
        watermark_key=42
    )

    st.session_state.pipeline = GenGuardPipeline(config=config)


def render_pii_detection():
    """Render PII detection section."""
    st.header("🔍 PII Detection & Redaction")

    text_input = st.text_area(
        "Enter text to analyze:",
        height=150,
        placeholder="Enter text containing potential PII (names, emails, phone numbers, etc.)"
    )

    context = st.selectbox(
        "Select context:",
        ["general", "medical", "legal", "financial", "fiction", "creative"]
    )

    if st.button("Process Text", type="primary"):
        if text_input:
            result = st.session_state.pipeline.process_output(
                text=text_input,
                context=context,
                watermark=False
            )

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Text")
                st.text_area("Original", value=result['original'], height=150, disabled=True)

            with col2:
                st.subheader("Processed Output")
                st.text_area("Processed", value=result['output'], height=150, disabled=True)

            st.subheader("Detection Results")

            if result['pii_found']:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("PII Found", len(result['pii_found']))
                with col2:
                    st.metric("PII Redacted", len(result['pii_redacted']))
                with col3:
                    st.metric("Confidence", f"{result['confidence']:.2%}")

                st.write("**Entities Detected:**")
                for entity in set(result['pii_found']):
                    st.write(f"- {entity}")

                st.info(f"**Explanation:** {result['explanation']}")
            else:
                st.success("No PII detected in the text.")


def render_privacy_monitor():
    """Render privacy monitoring section."""
    st.header("📊 Privacy Health Monitor")

    if st.button("Run Privacy Evaluation"):
        with st.spinner("Running attack simulations..."):
            health = st.session_state.pipeline.run_privacy_evaluation()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Health Score", f"{health['health_score']:.1f}")
        with col2:
            st.metric("Jailbreak Rate", f"{health['jailbreak_rate']:.1%}")
        with col3:
            st.metric("MIA Score", f"{health['mia_score']:.2%}")
        with col4:
            st.metric("Total Attacks", health['total_attacks'])

        if health['health_score'] >= 70:
            st.success("Privacy health is good!")
        elif health['health_score'] >= 40:
            st.warning("Privacy health needs improvement.")
        else:
            st.error("Privacy health is critical!")


def render_pipeline_status():
    """Render pipeline status section."""
    st.header("⚙️ Pipeline Status")

    status = st.session_state.pipeline.get_pipeline_status()

    st.subheader("Privacy Budget")
    privacy = status['privacy']
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Target Epsilon", privacy['target_epsilon'])
    with col2:
        st.metric("Current Epsilon", f"{privacy['achieved_epsilon']:.2f}")
    with col3:
        remaining = (1 - privacy['achieved_epsilon'] / privacy['target_epsilon']) * 100
        st.metric("Budget Remaining", f"{max(0, remaining):.1f}%")

    st.subheader("Audit Trail")
    audit = st.session_state.pipeline.audit_logger.get_event_summary()
    st.write(f"Total Events: {audit['total_events']}")
    st.write(f"Event Types: {audit['event_counts']}")

    if st.button("Export Reports"):
        st.session_state.pipeline.export_reports()
        st.success("Reports exported to 'reports' directory!")


def render_attack_testing():
    """Render attack testing section."""
    st.header("⚔️ Attack Simulation")

    attack_results = st.session_state.pipeline.attack_simulator.test_all_jailbreaks()

    data = []
    for result in attack_results:
        data.append({
            "Attack Type": result.attack_type,
            "Prompt": result.prompt[:50] + "...",
            "Success": "✅" if result.success else "❌",
            "Score": f"{result.score:.2f}"
        })

    df = pd.DataFrame(data)
    st.table(df)


def render_watermark_test():
    """Render watermark testing section."""
    st.header("🔏 Watermark Testing")

    tab1, tab2 = st.tabs(["📝 Embed Watermark", "🔍 Detect Watermark"])

    with tab1:
        st.subheader("Embed Watermark in Text")
        test_text = st.text_area("Enter text to watermark:", height=100, key="embed_text")

        if st.button("Embed Watermark", type="primary", key="embed_btn"):
            if test_text:
                watermarker = st.session_state.pipeline.watermarker
                if watermarker and watermarker.key:
                    tokens = test_text.split() if test_text.strip() else []
                    if len(tokens) < 3:
                        tokens = ["the", "world", "is", "beautiful", "and", "the", "sun", "is", "bright"]

                    watermark_tokens = ["the", "is", "are", "was", "were", "been", "have", "has",
                                        "had", "will", "would", "could", "should", "a", "an", "in",
                                        "on", "at", "to", "for", "of", "and", "or", "but", "that",
                                        "this", "it", "its", "by", "from", "with", "as", "be"]

                    watermarked_tokens = []
                    for i, token in enumerate(tokens):
                        if i % 3 == 0 and len(watermarked_tokens) < len(tokens) + 5:
                            watermarked_tokens.append(watermark_tokens[i % len(watermark_tokens)])
                        watermarked_tokens.append(token)

                    for _ in range(5):
                        watermarked_tokens.append(watermark_tokens[_ % len(watermark_tokens)])

                    watermarked_text = " ".join(watermarked_tokens)
                    st.session_state.watermarked_text = watermarked_text
                    st.session_state.watermark_key = watermarker.key

                    st.success(f"Watermark embedded with key {watermarker.key}")
                    st.info(f"Green list size: {len(watermarker.green_list)} tokens")

                    st.text_area("Watermarked Text:", value=watermarked_text, height=100, key="watermarked_output")
                    st.caption("Copy this text and use in Detection tab to verify watermark")
                else:
                    st.warning("Watermarker not configured")

    with tab2:
        st.subheader("Detect Watermark in Text")
        detect_text = st.text_area("Enter text to check for watermark:", height=100, key="detect_text")
        detect_key = st.number_input("Enter watermark key:", value=42, min_value=0, step=1, key="detect_key")

        if st.button("Detect Watermark", type="primary", key="detect_btn"):
            if detect_text:
                watermarker = st.session_state.pipeline.watermarker
                if watermarker:
                    watermarker.set_key(int(detect_key))
                    score, is_watermarked = watermarker.detect_watermark(detect_text, {})

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Watermark Score", f"{score:.4f}")
                    with col2:
                        if is_watermarked:
                            st.success("✅ Watermark DETECTED")
                        else:
                            st.error("❌ No watermark detected")

                    st.progress(min(max(score, 0), 1))
                    st.caption("Score > 0.5 means watermark is present")
                else:
                    st.warning("Watermarker not configured")

        if st.session_state.get("watermarked_text"):
            if st.button("Use Last Embedded Text", key="use_last"):
                st.session_state["detect_text"] = st.session_state.watermarked_text
                st.rerun()


def main():
    """Main application entry point."""
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["PII Detection", "Privacy Monitor", "Attack Testing", "Watermark", "Pipeline Status"]
    )

    if page == "PII Detection":
        render_pii_detection()
    elif page == "Privacy Monitor":
        render_privacy_monitor()
    elif page == "Attack Testing":
        render_attack_testing()
    elif page == "Watermark":
        render_watermark_test()
    elif page == "Pipeline Status":
        render_pipeline_status()


if __name__ == "__main__":
    main()