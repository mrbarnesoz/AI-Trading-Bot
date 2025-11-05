// Enable Bootstrap tooltips for any element with a title attribute.
document.addEventListener('DOMContentLoaded', function () {
  const tooltipTriggerList = [].slice.call(document.querySelectorAll('[title]'));
  tooltipTriggerList.forEach(function (el) {
    new bootstrap.Tooltip(el);
  });

  // Start log polling if the log element exists.
  const logElem = document.getElementById('logOutput');
  if (logElem) {
    const logsUrl = logElem.dataset.logsUrl || '/logs';
    const fetchLogs = () => {
      fetch(logsUrl)
        .then((response) => response.text())
        .then((text) => {
          logElem.textContent = text || 'No log output yet.';
          logElem.scrollTop = logElem.scrollHeight;
        })
        .catch((err) => console.error('Error fetching logs:', err));
    };
    fetchLogs();
    setInterval(fetchLogs, 2000);
  }

  // Initialize the trade history chart if data is available.
  const chartCanvas = document.getElementById('tradeChart');
  if (chartCanvas && typeof Chart !== 'undefined') {
    try {
      const rawVals = JSON.parse(chartCanvas.dataset.vals || '[]');
      const dataVals = Array.isArray(rawVals) ? rawVals.map((val) => Number(val) || 0) : [];
      const labels = dataVals.map((_, idx) => `Trade ${idx + 1}`);
      const colors = dataVals.map((val) => (val >= 0 ? 'rgba(25,135,84,0.6)' : 'rgba(220,53,69,0.6)'));
      const ctx = chartCanvas.getContext('2d');
      new Chart(ctx, {
        type: 'bar',
        data: {
          labels,
          datasets: [
            {
              label: 'P&L',
              data: dataVals,
              backgroundColor: colors,
            },
          ],
        },
        options: {
          plugins: {
            legend: { display: false },
          },
          scales: {
            x: { title: { display: true, text: 'Recent Trades' } },
            y: { title: { display: true, text: 'Profit / Loss' } },
          },
        },
      });
    } catch (err) {
      console.error('Unable to render trade chart:', err);
    }
  }
});

// Expose helper for capital slider label update.
function updateCapLabel(val) {
  const label = document.getElementById('capLabel');
  if (label) {
    label.innerText = `${val}%`;
  }
}

window.updateCapLabel = updateCapLabel;
