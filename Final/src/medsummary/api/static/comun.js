// Código compartido por las tres páginas: cabecera, llamadas a la API, tablas y el botón del health check.
const PAGINAS = [['/', 'Procesar diálogo'], ['/registros', 'Registros'], ['/metricas', 'Métricas']];

function crear(etiqueta, atributos = {}, texto = '') {
  const el = document.createElement(etiqueta);
  for (const [k, v] of Object.entries(atributos)) el.setAttribute(k, v);
  if (texto !== '') el.textContent = texto;
  return el;
}

async function api(ruta, opciones) {
  const r = await fetch(ruta, opciones);
  let datos = null;
  try { datos = await r.json(); } catch (e) { /* sin cuerpo JSON */ }
  if (!r.ok) {
    const error = new Error((datos && datos.detail) ? (typeof datos.detail === 'string' ? datos.detail : JSON.stringify(datos.detail)) : `Error ${r.status}`);
    error.datos = datos;
    throw error;
  }
  return datos;
}

function postJson(ruta, cuerpo) {
  return api(ruta, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(cuerpo) });
}

function numero(v, dec = 2) {
  return (v === null || v === undefined || Number.isNaN(v)) ? '-' : Number(v).toFixed(dec);
}

// columnas: [{nombre, numerica}] ; filas: listas de celdas (texto)
function tabla(columnas, filas) {
  const t = crear('table');
  const cab = crear('tr');
  columnas.forEach(c => cab.appendChild(crear('th', c.numerica ? { class: 'num' } : {}, c.nombre)));
  t.appendChild(crear('thead')).appendChild(cab);
  const cuerpo = crear('tbody');
  filas.forEach(fila => {
    const tr = crear('tr');
    fila.forEach((celda, i) => tr.appendChild(crear('td', columnas[i].numerica ? { class: 'num' } : {}, String(celda))));
    cuerpo.appendChild(tr);
  });
  t.appendChild(cuerpo);
  const contenedor = crear('div', { class: 'tabla' });
  contenedor.appendChild(t);
  return contenedor;
}

function construirCabecera(activa) {
  const barra = crear('header', { class: 'barra' });
  barra.appendChild(crear('h1', {}, 'MedSummary'));
  const nav = crear('nav');
  PAGINAS.forEach(([ruta, nombre]) => nav.appendChild(crear('a', { href: ruta, class: ruta === activa ? 'activo' : '' }, nombre)));
  barra.appendChild(nav);
  barra.appendChild(crear('div', { class: 'espacio' }));
  const boton = crear('button', { id: 'botonSalud', class: 'secundario', type: 'button' }, 'Verificar salud');
  barra.appendChild(boton);
  document.body.prepend(crear('div', { id: 'panelSalud' }));
  document.body.prepend(barra);
  boton.addEventListener('click', verificarSalud);
}

const NOMBRES_MODELOS = {
  modelo_ajustado: 'Modelo ajustado (Flan-T5 + LoRA)', embedder: 'Embedder (bge-small)', reranker: 'Reranker (cross-encoder)',
  bm25: 'Índice BM25', qwen: 'Qwen del agente (GGUF cuantizado)',
};

async function verificarSalud() {
  const boton = document.getElementById('botonSalud');
  const panel = document.getElementById('panelSalud');
  boton.disabled = true;
  const tarjeta = crear('div', { class: 'tarjeta' });
  tarjeta.appendChild(crear('h2', {}, 'Estado del sistema (GET /health)'));
  let datos = null;
  const inicio = performance.now();
  try {
    datos = await api('/health');
  } catch (error) {
    datos = error.datos;
    if (!datos) tarjeta.appendChild(crear('p', { class: 'error' }, `El sistema no respondió: ${error.message}`));
  }
  const ms = Math.round(performance.now() - inicio);
  if (datos && datos.modelos) {
    const listo = datos.status === 'ok';
    tarjeta.appendChild(crear('p', { class: listo ? 'ok' : 'error' },
      listo ? `Sistema listo (respondió en ${ms} ms, sin ejecutar ningún modelo).`
            : `No está listo: ${datos.carga.detalle}`));
    const marca = v => v ? '✓ cargado' : '✗ no cargado';
    const filas = Object.entries(datos.modelos).map(([m, v]) => [NOMBRES_MODELOS[m] || m, marca(v)]);
    const indice = datos.indice || {};
    filas.push(['Índice vectorial (ChromaDB)', indice.responde ? `✓ responde (${indice.chunks} chunks)` : '✗ no responde']);
    tarjeta.appendChild(tabla([{ nombre: 'Componente' }, { nombre: 'Estado' }], filas));
    const c = datos.carga;
    tarjeta.appendChild(crear('p', { class: 'suave' },
      `Dispositivo: ${c.dispositivo || '-'} · Cuantización de Qwen: ${datos.cuantizacion}` + (c.segundos ? ` · Carga inicial: ${c.segundos} s` : '')));
  }
  const cerrar = crear('button', { class: 'secundario', type: 'button' }, 'Cerrar');
  cerrar.addEventListener('click', () => panel.replaceChildren());
  tarjeta.appendChild(cerrar);
  const contenedor = crear('main');
  contenedor.appendChild(tarjeta);
  panel.replaceChildren(contenedor);
  boton.disabled = false;
}
