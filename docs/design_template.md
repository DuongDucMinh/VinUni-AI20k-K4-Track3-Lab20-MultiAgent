# Multi-Agent Research System Design Document

## Problem

Hệ thống cần xử lý các yêu cầu nghiên cứu kỹ thuật phức tạp (ví dụ: so sánh kiến trúc Multi-Agent, phân tích kỹ thuật phân rã tác vụ, cơ chế kiểm soát ảo giác) đòi hỏi thu thập dữ liệu nguồn đáng tin cậy, trích xuất dẫn chứng chính xác, phân tích các góc nhìn đa chiều, và tổng hợp thành báo cáo hoàn chỉnh có trích dẫn khoa học (citations).

## Why Multi-Agent?

- **Phân tách trách nhiệm (Separation of Concerns):** Single-agent dễ gặp hiện tượng quá tải ngữ cảnh (context bloat), bỏ sót dữ liệu hoặc tự tạo dẫn chứng ảo (hallucinations) khi phải vừa tìm kiếm, vừa phản biện, vừa viết văn bản dài trong 1 lượt prompt duy nhất.
- **Kiểm soát quy trình (Deterministic Orchestration):** Multi-agent cho phép Supervisor định tuyến tuần tự và có điều kiện qua từng chặng: Thu thập bằng chứng (`Researcher`) -> Phân tích & Phát hiện mâu thuẫn (`Analyst`) -> Tổng hợp & Trích dẫn (`Writer`) -> Đánh giá & Chấm điểm độc lập (`Critic`).
- **Gia tăng độ tin cậy (Verifiability & Grounding):** Từng tuyên bố (claim) được gắn chặt với mã nguồn tài liệu `[source_id]`, giúp hệ thống đạt độ phủ trích dẫn cao và minh bạch.

## Agent Roles

| Agent | Responsibility | Input | Output | Failure Mode & Mitigation |
|---|---|---|---|---|
| **Supervisor** | Điều phối định tuyến trạng thái workflow, thực thi guardrails dừng và fallback | `ResearchState` | Cập nhật `route_history`, quyết định node tiếp theo | *Loop vô tận / kẹt route:* Giới hạn `max_iterations=6`, tự động chuyển về `writer` hoặc `done`. |
| **Researcher** | Tìm kiếm dữ liệu trong corpus offline (TF-IDF), thu thập `SourceDocument` và tạo research notes | `state.request.query`, `state.request.max_sources` | `state.sources`, `state.research_notes` | *Không tìm thấy tài liệu:* Trả về thông báo cảnh báo và fallback tóm tắt tổng quan. |
| **Analyst** | Phân tích sâu các luận điểm, so sánh trade-offs, phát hiện mâu thuẫn và lỗ hổng bằng chứng | `state.research_notes` | `state.analysis_notes` | *Phân tích sơ sài:* Sử dụng cấu trúc phân tích 4 phần bắt buộc trong system prompt. |
| **Writer** | Tổng hợp báo cáo nghiên cứu kỹ thuật hoàn chỉnh, gắn trích dẫn `[source_id]` chuẩn xác | `research_notes`, `analysis_notes`, `sources` | `state.final_answer`, `citation_ids_used` | *Ảo giác / thiếu citation:* Yêu cầu cú pháp `[source_id]` bắt buộc và tự động trích xuất danh sách citation. |
| **Critic** | Phản biện độc lập, kiểm tra độ chính xác sự thật, phát hiện ảo giác và chấm điểm chất lượng (0-10) | `final_answer`, `sources` | `quality_score`, `critique` | *LLM lỗi:* Fallback chấm điểm baseline và ghi nhận cảnh báo vào trace. |

## Shared State

- `request` (`ResearchQuery`): Truy vấn gốc từ người dùng, số nguồn tối đa, đối tượng độc giả mục tiêu.
- `iteration` (`int`): Bộ đếm số vòng lặp workflow để ngăn chặn chạy vô hạn.
- `route_history` (`list[str]`): Lịch sử chuyển tiếp giữa các Agent, dùng để debug và routing.
- `sources` (`list[SourceDocument]`): Danh sách các tài liệu / trích đoạn đã thu thập được từ corpus.
- `research_notes` (`str | None`): Ghi chú thô có dẫn chứng từ Researcher.
- `analysis_notes` (`str | None`): Báo cáo phân tích cấu trúc, trade-offs từ Analyst.
- `final_answer` (`str | None`): Báo cáo nghiên cứu tổng hợp hoàn chỉnh từ Writer.
- `agent_results` (`list[AgentResult]`): Chi tiết kết quả thực thi, metadata, tokens và độ trễ của từng agent.
- `token_usage` (`dict[str, int]`): Tổng số input tokens và output tokens tích lũy của toàn bộ workflow.
- `total_cost_usd` (`float`): Chi phí API ước tính tích lũy (USD).
- `citation_ids_used` (`list[str]`): Danh sách các ID nguồn được trích dẫn trong báo cáo cuối.
- `agent_durations` (`dict[str, float]`): Thời gian thực thi (giây) chi tiết cho từng agent.
- `trace` (`list[dict]`): Chuỗi sự kiện có gắn nhãn thời gian để xuất trace log ra LangSmith và JSON.
- `errors` (`list[str]`): Danh sách lỗi thu thập được trong quá trình chạy.

## Routing Policy

Graph được xây dựng bằng `LangGraph.StateGraph` theo cơ chế chu trình tuần tự có điều phối:

```text
[START]
   │
   ▼
[Supervisor] ──(research_notes is None)──────────► [Researcher] ──┐
   ▲  │                                                           │
   │  ├──(analysis_notes is None)────────► [Analyst] ─────────────┤
   │  │                                                           │
   │  ├──(final_answer is None)──────────► [Writer] ──────────────┤
   │  │                                                           │
   │  ├──(include_critic & unverified)───► [Critic] ──────────────┤
   │  │                                                           │
   │  └───────────────────────────────────────────────────────────┘
   │
   └──(final_answer exists OR iteration >= max_iterations)──► [END]
```

## Guardrails

- **Max Iterations:** Cấu hình qua `MAX_ITERATIONS=6`. Supervisor tự động ép dừng và chuyển `ROUTE_DONE` nếu vượt quá.
- **Timeout:** Cấu hình qua `TIMEOUT_SECONDS=60` trên OpenAI/Groq client call và workflow invoke.
- **Retry & Backoff:** Sử dụng thư viện `tenacity` với exponential backoff (`min=2s, max=6s`) cho mọi lệnh gọi LLM.
- **Model Fallback:** Hỗ trợ danh sách fallback models tự động (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`, `mixtral-8x7b-32768`) khi gặp lỗi mã 429 (Rate Limit).
- **Rate Limit Pacing:** Tự động giãn cách (`inter_call_delay_seconds=1.5s`) giữa các lượt gọi LLM để không vượt quá giới hạn 30 RPM / 8K TPM của Groq.
- **Validation:** Tất cả dữ liệu truyền qua các Node đều tuân thủ nghiêm ngặt Pydantic Schemas (`ResearchState`, `ResearchQuery`, `SourceDocument`, `AgentResult`).

## Benchmark Plan

- **Queries thử nghiệm:** 3 câu hỏi nghiên cứu về Multi-Agent Systems từ corpus `ai_agent_offline_research_corpus_v2`.
- **Metrics so sánh:**
  1. `Latency (s)`: Wall-clock time từ lúc nhận query đến khi xuất kết quả.
  2. `Cost (USD)`: Chi phí token theo biểu giá mô hình.
  3. `Quality Score`: Thang điểm 0-10 do Critic Agent đánh giá dựa trên tiêu chuẩn rubric.
  4. `Citation Coverage`: Tỷ lệ phần trăm nguồn dữ liệu được trích dẫn hợp lệ `[source_id]`.
  5. `Failure Rate`: Tỷ lệ lỗi gặp phải trên tổng số lượt xử lý.

