mkdir -p ~/.streamlit/
echo "\
[theme]\n\
primaryColor=’#020202’\n\
backgroundColor=’#c4c3c3’\n\
secondaryBackgroundColor=’#ebd316’\n\
font = ‘sans serif’\n\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
