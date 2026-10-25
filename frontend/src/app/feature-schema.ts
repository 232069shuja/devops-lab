export type Feature = {
  name: string;
  label: string;
  kind: 'text' | 'number' | 'category';
  placeholder?: string;
  required: boolean;
  options?: string[];
  min?: number;
  max?: number;
};

// Raw input feature schema matching backend model bundle
export const FEATURES: Feature[] = [
  {
    name: 'comment_text',
    label: 'Comment Text',
    kind: 'text',
    placeholder: 'Enter a comment to evaluate for toxicity (e.g., "I love the idea of upvoting entire articles.").',
    required: true,
  },
];
