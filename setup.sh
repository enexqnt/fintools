mkdir -p ~/.streamlit/
echo "[theme]
primaryColor=’#020202’
backgroundColor=’#c4c3c3’
secondaryBackgroundColor=’#ebd316’
font = ‘sans serif’
[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
