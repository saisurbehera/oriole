#!/bin/bash
# Example curl commands for NetHack KGQA API

echo "🎮 NetHack KGQA API - Curl Examples"
echo "="*50

API_URL="http://localhost:8000"

echo -e "\n1. Check service status:"
echo "curl $API_URL/"

echo -e "\n2. Health check:"
echo "curl $API_URL/health"

echo -e "\n3. Ask about Medusa:"
echo 'curl -X POST "$API_URL/query" \
  -H "Content-Type: application/json" \
  -d "{
    \"question\": \"What is Medusa'\''s armor class and how dangerous is she?\",
    \"max_facts\": 5
  }"'

echo -e "\n4. Ask about combat strategy:"
echo 'curl -X POST "$API_URL/query" \
  -H "Content-Type: application/json" \
  -d "{
    \"question\": \"How should I fight a dragon?\",
    \"entity\": \"Dragon\",
    \"include_predictions\": true
  }"'

echo -e "\n5. Ask about eating corpses:"
echo 'curl -X POST "$API_URL/query" \
  -H "Content-Type: application/json" \
  -d "{
    \"question\": \"What happens if I eat a floating eye corpse?\",
    \"max_facts\": 3
  }"'

echo -e "\n6. List top entities:"
echo "curl $API_URL/entities"

echo -e "\n7. List relation types:"
echo "curl $API_URL/relations"

echo -e "\n8. Get specific entity info:"
echo "curl $API_URL/entity/Soldier%20ant"

echo -e "\n9. Speed comparison query:"
echo 'curl -X POST "$API_URL/query" \
  -H "Content-Type: application/json" \
  -d "{
    \"question\": \"Which monsters are very fast?\",
    \"max_facts\": 8
  }"'

echo -e "\n10. Resistance planning:"
echo 'curl -X POST "$API_URL/query" \
  -H "Content-Type: application/json" \
  -d "{
    \"question\": \"What resistances should I prioritize?\",
    \"max_facts\": 6
  }"'

echo -e "\nTo run a specific example, copy the curl command above!"
echo -e "\nTo start the server: ./start_server.sh"