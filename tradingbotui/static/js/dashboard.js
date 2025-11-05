(function () {
  const form = document.getElementById('execution-form');
  if (form) {
    const actionField = document.getElementById('execution-action');
    form.querySelectorAll('button[data-action]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const action = btn.getAttribute('data-action');
        if (action === 'live' && !confirm('This will execute live orders. Proceed?')) {
          return;
        }
        if (actionField) {
          actionField.value = action;
        }
        form.submit();
      });
    });
  }

  const slider = document.getElementById('capitalRange');
  if (slider) {
    if (typeof window.updateCapLabel === 'function') {
      window.updateCapLabel(slider.value);
    }
    slider.addEventListener('input', () => {
      const hidden = document.querySelector('#retrain-form input[name="capital_pct"]');
      if (hidden) {
        hidden.value = slider.value;
      }
      if (typeof window.updateCapLabel === 'function') {
        window.updateCapLabel(slider.value);
      }
    });
  }

  const retrainForm = document.getElementById('retrain-form');
  if (retrainForm) {
    const syncRetrainFields = () => {
      const strategySelect = document.getElementById('strategySelect');
      const pairSelect = document.getElementById('pairSelect');
      const tfSelect = document.getElementById('tfSelect');
      const strategyField = retrainForm.querySelector('input[name="strategy"]');
      const pairField = retrainForm.querySelector('input[name="pair"]');
      const tfField = retrainForm.querySelector('input[name="timeframe"]');
      if (strategySelect && strategyField) {
        strategyField.value = strategySelect.value;
      }
      if (pairSelect && pairField) {
        const selected = Array.from(pairSelect.selectedOptions);
        pairField.value = selected.length ? selected[0].value : pairSelect.options.length ? pairSelect.options[0].value : '';
      }
      if (tfSelect && tfField) {
        tfField.value = tfSelect.value;
      }
      const capitalHidden = retrainForm.querySelector('input[name="capital_pct"]');
      if (capitalHidden && slider) {
        capitalHidden.value = slider.value;
      }
    };

    ['change', 'input'].forEach((evt) => {
      const strategySelect = document.getElementById('strategySelect');
      const pairSelect = document.getElementById('pairSelect');
      const tfSelect = document.getElementById('tfSelect');
      if (strategySelect) {
        strategySelect.addEventListener(evt, syncRetrainFields);
      }
      if (pairSelect) {
        pairSelect.addEventListener(evt, syncRetrainFields);
      }
      if (tfSelect) {
        tfSelect.addEventListener(evt, syncRetrainFields);
      }
    });
    syncRetrainFields();

    retrainForm.addEventListener('submit', (event) => {
      const confirmBtn = retrainForm.querySelector('button[data-confirm]');
      const message = confirmBtn ? confirmBtn.getAttribute('data-confirm') : null;
      if (message && !confirm(message)) {
        event.preventDefault();
      }
    });
  }

  const diffOutput = document.getElementById('diffOutput');
  document.querySelectorAll('button[data-diff]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const config = btn.getAttribute('data-diff');
      if (!config || !diffOutput) {
        return;
      }
      diffOutput.textContent = 'Loading diff...';
      try {
        const response = await fetch('/filemanager/diff?config=' + encodeURIComponent(config));
        diffOutput.textContent = await response.text();
      } catch (err) {
        diffOutput.textContent = 'Failed to load diff: ' + err;
      }
    });
  });

  if (typeof Chart !== 'undefined') {
    const equityCanvas = document.getElementById('equityChart');
    if (equityCanvas) {
      let series = [];
      try {
        series = JSON.parse(equityCanvas.dataset.series || '[]');
      } catch (err) {
        series = [];
      }
      new Chart(equityCanvas.getContext('2d'), {
        type: 'line',
        data: {
          labels: series.map((_, idx) => idx + 1),
          datasets: [{
            label: 'Equity',
            data: series,
            borderColor: '#0d6efd',
            backgroundColor: 'rgba(13,110,253,0.1)',
            tension: 0.25,
            fill: true,
          }],
        },
        options: { scales: { x: { display: false } } },
      });
    }

    const drawdownCanvas = document.getElementById('drawdownChart');
    if (drawdownCanvas) {
      let series = [];
      try {
        series = JSON.parse(drawdownCanvas.dataset.series || '[]');
      } catch (err) {
        series = [];
      }
      new Chart(drawdownCanvas.getContext('2d'), {
        type: 'line',
        data: {
          labels: series.map((_, idx) => idx + 1),
          datasets: [{
            label: 'Drawdown',
            data: series,
            borderColor: '#dc3545',
            backgroundColor: 'rgba(220,53,69,0.1)',
            tension: 0.25,
            fill: true,
          }],
        },
        options: { scales: { x: { display: false } } },
      });
    }
  }

  const copyButton = document.getElementById('copyErrors');
  if (copyButton) {
    copyButton.addEventListener('click', async () => {
      const target = document.getElementById('errorLogText');
      if (!target) {
        return;
      }
      try {
        await navigator.clipboard.writeText(target.textContent || '');
        copyButton.textContent = 'Copied!';
        setTimeout(() => (copyButton.textContent = 'Copy to Clipboard'), 1500);
      } catch (err) {
        copyButton.textContent = 'Copy failed';
        setTimeout(() => (copyButton.textContent = 'Copy to Clipboard'), 1500);
      }
    });
  }
})();
