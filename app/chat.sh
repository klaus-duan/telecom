curl -i \
     -X POST http://127.0.0.1:8000/chat \
     -H 'content-type: application/json' \
     -d '{"request_id":"r1","message":"这个月你主推的套餐"}'
