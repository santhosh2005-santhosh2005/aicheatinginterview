
const http = require('http');

const options = {
    hostname: 'localhost',
    port: 11434,
    path: '/api/tags',
    method: 'GET'
};

const req = http.request(options, (res) => {
    let data = '';
    res.on('data', (chunk) => {
        data += chunk;
    });
    res.on('end', () => {
        try {
            const models = JSON.parse(data).models;
            console.log('Available models:', models.map(m => m.name));
        } catch (e) {
            console.error('Error parsing JSON:', e);
        }
    });
});

req.on('error', (e) => {
    console.error(`Problem with request: ${e.message}`);
});

req.end();
