const colorPink = new Color('#CF4D6F');
const colorWhite = new Color('#FFFFFF');

const colormap = colorWhite.range(colorPink, {space: "srgb-linear"});

/**
 * Mostra o link da página atual ativado.
 */
function activateLink() {
    let path= window.location.pathname;
    let page= path.split("/").pop();
    let element = null;
    if(page.length === 0) {
        element = document.getElementById('nav-link-analysis');
    } else {
        element = document.getElementById('nav-link-' + page);
    }
    if(element != null) {
        element.classList.add('active');
    }
}

/**
 * Mostra o spinner e oculta a tabela eo texto de resposta.
 */
function showSpinner() {
    document.getElementById('table-spinner').style.display = 'block';
    document.getElementById('div-answer-table').style.display = 'none';
    document.getElementById('div-answer-text').style.display = 'none';
}

/**
 * Oculta o spinner e mostra a tabela e o texto de resposta.
 */
function hideSpinner() {
    document.getElementById('table-spinner').style.display = 'none';
    document.getElementById('div-answer-table').style.display = 'block';
    document.getElementById('div-answer-text').style.display = 'block';
}

function normalize(value) {
    return Math.max(0, value);
}

/**
 * Constrói o texto de visualização da página principal, colorindo quais palavras mais contribuíram para a predição
 */
function buildText(visualization) {
    let div_answer_text = document.getElementById("div-answer-text");
    while (div_answer_text.firstChild) {
        div_answer_text.removeChild(div_answer_text.lastChild);
    }

    let paragraph = document.createElement('p');

    let add_padding = false;
    for(let i = 0; i < visualization['text_tokens'].length; i++) {
        let span = document.createElement('span');
        let token = visualization['text_tokens'][i];
        let pad = add_padding && (token[0] !== '#')? ' ' : '';

        add_padding = token[token.length - 1] !== '#';
        console.log(token, add_padding);

        span.innerText = pad + token.replaceAll('#', '');
        span.style = 'background-color: ' + colormap(normalize(visualization['word_attention'][i]));
        paragraph.appendChild(span);
    }
    div_answer_text.appendChild(paragraph);
}

/**
 * Constrói a tabela, onde na primeira coluna estão os rótulos e na segunda a probabilidade de que o texto fornecido
 * pertença àquela classe
 */
function buildTable(prediction) {
    let div_answer_table = document.getElementById("div-answer-table");
    while (div_answer_table.firstChild) {
        div_answer_table.removeChild(div_answer_table.lastChild);
    }

    let table = document.createElement('table');
    table.setAttribute("id", "answer_table");
    table.classList.add("sentiment");

    let thead = table.createTHead();
    let tbody = table.createTBody();

    for(let i = 0; i < prediction.length; i++) {
        // constrói o cabeçalho
        if(i === 0) {
            let tr = thead.insertRow();
            for(let key in prediction[i]) {
                let th = document.createElement("th");
                th.innerHTML = key;
                tr.appendChild(th);
            }
            thead.appendChild(tr);
        }
        // constrói o resto
        let tr = tbody.insertRow();
        for(let key in prediction[i]) {
            let td = tr.insertCell();

            let value = prediction[i][key];
            if(!isNaN(parseFloat(value))) {
                value = parseFloat(value).toFixed(4);
            }

            td.appendChild(document.createTextNode(value));
        }
    }
    div_answer_table.appendChild(table);
}
