# Lab Guide: Multi-Agent Research System

## Scenario

Bạn cần xây dựng một research assistant có thể nhận câu hỏi dài, tìm thông tin, phân tích và viết câu trả lời cuối cùng. Lab yêu cầu so sánh hai cách làm:

1. **Single-agent baseline**: một agent làm toàn bộ.
2. **Multi-agent workflow**: Supervisor điều phối Researcher, Analyst, Writer.

## Quy tắc quan trọng

- Không thêm agent nếu không có lý do rõ ràng.
- Mỗi agent phải có responsibility riêng.
- Shared state phải đủ rõ để debug.
- Phải có trace hoặc log cho từng bước.
- Phải benchmark, không chỉ nhìn output bằng cảm tính.

## Milestone 1: Baseline

File gợi ý:

- `src/multi_agent_research_lab/cli.py`
- `src/multi_agent_research_lab/services/llm_client.py`

TODO(student): thay baseline placeholder bằng một call LLM thật.

## Milestone 2: Supervisor

File gợi ý:

- `src/multi_agent_research_lab/agents/supervisor.py`
- `src/multi_agent_research_lab/graph/workflow.py`

TODO(student): implement routing policy.

Gợi ý câu hỏi thiết kế:

- Khi nào gọi Researcher?
- Khi nào gọi Analyst?
- Khi nào gọi Writer?
- Khi nào stop?
- Nếu agent fail thì retry hay fallback?

## Milestone 3: Worker agents

File gợi ý:

- `src/multi_agent_research_lab/agents/researcher.py`
- `src/multi_agent_research_lab/agents/analyst.py`
- `src/multi_agent_research_lab/agents/writer.py`

TODO(student): implement từng worker.

## Milestone 4: Trace và benchmark

File gợi ý:

- `src/multi_agent_research_lab/observability/tracing.py`
- `src/multi_agent_research_lab/evaluation/benchmark.py`
- `src/multi_agent_research_lab/evaluation/report.py`

Benchmark tối thiểu:

| Metric | Cách đo gợi ý |
|---|---|
| Latency | wall-clock time |
| Cost | token usage hoặc provider usage |
| Quality | rubric 0-10 do peer review |
| Citation coverage | số claims có source / tổng claims chính |
| Failure rate | số query fail / tổng query |

## Troubleshooting

### macOS: lỗi SSL certificate khi gọi API qua HTTPS (Tavily, OpenAI, ...)

Triệu chứng: khi implement `SearchClient` (hoặc bất kỳ HTTPS call nào) trên macOS, bạn có thể gặp lỗi kiểu:

```
ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed:
unable to get local issuer certificate
```

Nguyên nhân: Python cài từ python.org trên macOS **không dùng** certificate store của hệ điều hành, nên không tìm thấy CA bundle hợp lệ. Đây là lỗi môi trường, **không phải** do API key sai.

Cách khắc phục (chọn 1 trong 3):

1. **Chạy script cài certificate đi kèm Python** (nhanh nhất):

   ```bash
   /Applications/Python\ 3.12/Install\ Certificates.command
   ```

   (thay `3.12` bằng version Python của bạn)

2. **Dùng `certifi` trong code** — thêm `certifi` vào dependencies, rồi tạo SSL context khi gọi HTTPS:

   ```python
   import certifi
   import ssl
   from urllib.request import urlopen

   ssl_context = ssl.create_default_context(cafile=certifi.where())
   urlopen(request, timeout=timeout, context=ssl_context)
   ```

3. **Set biến môi trường** trỏ tới CA bundle của certifi (không cần đổi code):

   ```bash
   export SSL_CERT_FILE=$(python -m certifi)
   ```

## Exit ticket

### 1. Case nào nên dùng multi-agent? Vì sao?
- **Các tác vụ phức tạp, phân rã được theo giai đoạn chuyên môn (Decoupled Sub-tasks):** Ví dụ như nghiên cứu kỹ thuật đa chặng (Thu thập dữ liệu -> Phân tích lỗ hổng/Mâu thuẫn -> Viết báo cáo khoa học -> Phản biện kiểm chứng thực tế).
- **Yêu cầu phân tách quyền hạn và ngữ cảnh (Context Isolation & Least Privilege):** Khi mỗi bước chỉ cần một tập dữ liệu hoặc công cụ đặc thù (ngăn chặn hiện tượng quá tải context window và rò rỉ dữ liệu).
- **Cần cơ chế kiểm tra chéo độc lập (Verification & Guardrailing):** Khi cần một agent độc lập (như Critic/Verifier) để fact-check và gắn mã citation mà không bị thiên kiến tự xác nhận (confirmation bias) từ agent tạo sinh.

### 2. Case nào không nên dùng multi-agent? Vì sao?
- **Các tác vụ đơn giản, phản hồi trực tiếp (Single-turn Q&A, Tóm tắt ngắn, Phân loại văn bản):** Vì hệ thống multi-agent sẽ tạo ra độ trễ cao không cần thiết (high latency) và tiêu tốn token gấp nhiều lần (higher cost).
- **Tác vụ có tính phụ thuộc tuần tự tuyến tính nghiêm ngặt không cần feedback loop:** Khi một pipeline deterministic đơn giản (prompt chaining / DSPy pipeline) đã đủ giải quyết mà không cần điều phối linh hoạt.
- **Môi trường yêu cầu độ trễ cực thấp (Sub-second Latency SLA):** Việc chuyển giao trạng thái (state handoffs) và tuần tự gọi nhiều lượt LLM sẽ làm vi phạm yêu cầu thời gian thực.

