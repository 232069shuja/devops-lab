import { Component, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { FEATURES, Feature } from './feature-schema';
import { PredictionService, PredictionResponse } from './prediction.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  readonly fields = FEATURES;
  readonly isStaticDeployment = !['localhost', '127.0.0.1'].includes(window.location.hostname);

  values: Record<string, string | number | null> = {};
  loading = false;
  error = '';
  result: PredictionResponse | null = null;

  samplePresets = [
    {
      title: 'Civil Comment (Benign)',
      text: 'I love the idea of upvoting entire articles and sharing thoughtful feedback with the community.',
    },
    {
      title: 'Identity-Mention (Unintended Bias Test)',
      text: 'As a proud Muslim woman, I volunteer regularly at our local food pantry and library.',
    },
    {
      title: 'Toxic Attack (High Toxicity)',
      text: 'haha you guys are a bunch of losers who do not know anything.',
    },
  ];

  private api = inject(PredictionService);

  constructor() {
    this.resetForm();
  }

  resetForm() {
    for (const field of this.fields) {
      this.values[field.name] = '';
    }
    // Load default sample
    this.values['comment_text'] = this.samplePresets[0].text;
  }

  loadPreset(preset: { title: string; text: string }) {
    this.values['comment_text'] = preset.text;
    this.error = '';
    this.result = null;
  }

  submit() {
    this.error = '';
    this.result = null;

    if (this.isStaticDeployment) {
      this.error = 'Static GitHub Pages Demonstration: Start the Flask API and Angular frontend locally for live machine learning inference.';
      return;
    }

    const payload: Record<string, string | number> = {};
    for (const field of this.fields) {
      const val = this.values[field.name];
      if (val === null || val === undefined || String(val).trim() === '') {
        this.error = `Please provide a valid entry for "${field.label}".`;
        return;
      }
      if (field.kind === 'number') {
        const num = Number(val);
        if (!Number.isFinite(num)) {
          this.error = `Invalid numeric input for ${field.label}`;
          return;
        }
        payload[field.name] = num;
      } else {
        payload[field.name] = String(val).trim();
      }
    }

    this.loading = true;
    this.api.predict(payload).subscribe({
      next: (res) => {
        this.result = res;
        this.loading = false;
      },
      error: (err) => {
        this.error =
          `Inference request failed (Status ${err.status || 'Offline'}): ` +
          (err.error?.error || err.message || 'Make sure the Flask API is running on http://127.0.0.1:5000');
        this.loading = false;
      },
    });
  }
}
