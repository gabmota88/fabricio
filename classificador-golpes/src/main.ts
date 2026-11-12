
import "./style.css";

// ==========================================
// 1. TIPOS DO SISTEMA
// ==========================================

type Label = "GOLPE" | "LEGITIMO";

interface TrainingExample {
  text: string;
  label: Label;
}

interface Neighbor {
  text: string;
  label: Label;
  distance: number;
}

interface ClassificationResult {
  label: Label;
  confidence: number;
  neighbors: Neighbor[];
  tokens: string[];
}

// ==========================================
// 2. PRÉ-PROCESSAMENTO DE TEXTO
// ==========================================

class TextPreprocessor {

  private stopWords: Set<string> = new Set([
    "a", "o", "as", "os",
    "de", "do", "da", "dos", "das",
    "e", "em", "no", "na", "nos", "nas",
    "um", "uma", "para", "por",
    "que", "com", "se", "sua", "seu",
    "meu", "minha", "ao", "aos",
    "foi", "ser", "estar"
  ]);

  public normalize(text: string): string {

    return text
      .toLowerCase()
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "")
      .replace(/[^a-z0-9\s]/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  }

  public tokenize(text: string): string[] {

    const normalized = this.normalize(text);

    return normalized
      .split(" ")
      .filter(word => word.length > 1)
      .filter(word => !this.stopWords.has(word))
      .map(word => this.stem(word));
  }

  // Stemização simplificada para português.
  // Não substitui um algoritmo linguístico completo.

  private stem(word: string): string {

    const suffixes = [
      "amentos",
      "imentos",
      "amento",
      "imento",
      "acoes",
      "adores",
      "adoras",
      "idades",
      "mente",
      "acao",
      "ador",
      "adora",
      "idade",
      "ando",
      "endo",
      "indo",
      "ados",
      "adas",
      "idos",
      "idas",
      "ado",
      "ada",
      "ido",
      "ida",
      "es",
      "s"
    ];

    for (const suffix of suffixes) {

      if (
        word.endsWith(suffix) &&
        word.length - suffix.length >= 3
      ) {
        return word.slice(0, -suffix.length);
      }
    }

    return word;
  }
}

// ==========================================
// 3. BAG OF WORDS
// ==========================================

class BagOfWords {

  private vocabulary: string[] = [];

  private vocabularyIndex: Map<string, number> = new Map();

  public fit(documents: string[][]): void {

    const words = new Set<string>();

    documents.forEach(document => {

      document.forEach(word => {
        words.add(word);
      });

    });

    this.vocabulary = Array.from(words).sort();

    this.vocabularyIndex.clear();

    this.vocabulary.forEach((word, index) => {

      this.vocabularyIndex.set(word, index);

    });
  }

  public transform(tokens: string[]): number[] {

    const vector = new Array(
      this.vocabulary.length
    ).fill(0);

    tokens.forEach(word => {

      const index = this.vocabularyIndex.get(word);

      if (index !== undefined) {
        vector[index]++;
      }

    });

    return vector;
  }

  public getVocabulary(): string[] {

    return [...this.vocabulary];
  }
}

// ==========================================
// 4. CLASSIFICADOR KNN
// ==========================================

class KNNClassifier {

  private trainingVectors: number[][] = [];

  private trainingLabels: Label[] = [];

  private trainingTexts: string[] = [];

  private k: number;

  constructor(k: number = 3) {
    this.k = k;
  }

  public fit(
    vectors: number[][],
    labels: Label[],
    texts: string[]
  ): void {

    this.trainingVectors = vectors;

    this.trainingLabels = labels;

    this.trainingTexts = texts;
  }

  private cosineDistance(
    vectorA: number[],
    vectorB: number[]
  ): number {

    let dotProduct = 0;

    let normA = 0;

    let normB = 0;

    for (let i = 0; i < vectorA.length; i++) {

      dotProduct += vectorA[i] * vectorB[i];

      normA += vectorA[i] ** 2;

      normB += vectorB[i] ** 2;
    }

    const denominator = Math.sqrt(normA) * Math.sqrt(normB);

    // Caso os vetores sejam vazios.

    if (denominator === 0) {
      return 1;
    }

    const similarity = dotProduct / denominator;

    return 1 - similarity;
  }

  public predict(
    inputVector: number[]
  ): ClassificationResult {

    const neighbors: Neighbor[] = [];

    this.trainingVectors.forEach(
      (trainingVector, index) => {

        const distance = this.cosineDistance(
          inputVector,
          trainingVector
        );

        neighbors.push({
          text: this.trainingTexts[index],
          label: this.trainingLabels[index],
          distance
        });

      }
    );

    // Ordena do mais próximo para o mais distante.

    neighbors.sort(
      (a, b) => a.distance - b.distance
    );

    const nearestNeighbors = neighbors.slice(
      0,
      Math.min(this.k, neighbors.length)
    );

    const votes: Record<Label, number> = {
      GOLPE: 0,
      LEGITIMO: 0
    };

    nearestNeighbors.forEach(neighbor => {

      votes[neighbor.label]++;

    });

    const label: Label =
      votes.GOLPE > votes.LEGITIMO
        ? "GOLPE"
        : "LEGITIMO";

    const confidence =
      nearestNeighbors.length > 0
        ? votes[label] / nearestNeighbors.length
        : 0;

    return {
      label,
      confidence,
      neighbors: nearestNeighbors,
      tokens: []
    };
  }
}

// ==========================================
// 5. BASE DE TREINAMENTO
// ==========================================

const trainingData: TrainingExample[] = [

  {
    text: "Ganhe dinheiro agora clique no link",
    label: "GOLPE"
  },

  {
    text: "Sua conta bancaria sera bloqueada atualize seus dados",
    label: "GOLPE"
  },

  {
    text: "Parabens voce ganhou um premio resgate agora",
    label: "GOLPE"
  },

  {
    text: "Confirme sua senha para evitar o bloqueio da conta",
    label: "GOLPE"
  },

  {
    text: "Sua encomenda esta retida pague a taxa para liberar",
    label: "GOLPE"
  },

  {
    text: "Urgente transfira o dinheiro para esta conta",
    label: "GOLPE"
  },

  {
    text: "Seu CPF foi bloqueado regularize imediatamente",
    label: "GOLPE"
  },

  {
    text: "Clique no link e informe seus dados bancarios",
    label: "GOLPE"
  },

  {
    text: "Voce foi selecionado para receber um premio",
    label: "GOLPE"
  },

  {
    text: "Atualize seu cadastro para evitar o cancelamento",
    label: "GOLPE"
  },

  {
    text: "A reuniao sera realizada amanha as dez horas",
    label: "LEGITIMO"
  },

  {
    text: "Segue o relatorio solicitado para sua analise",
    label: "LEGITIMO"
  },

  {
    text: "Bom dia gostaria de confirmar nossa reuniao",
    label: "LEGITIMO"
  },

  {
    text: "O pagamento foi realizado conforme combinado",
    label: "LEGITIMO"
  },

  {
    text: "Encaminho os documentos solicitados por email",
    label: "LEGITIMO"
  },

  {
    text: "A manutencao do equipamento foi concluida",
    label: "LEGITIMO"
  },

  {
    text: "Obrigado pelo retorno entrarei em contato",
    label: "LEGITIMO"
  },

  {
    text: "Segue o arquivo com os resultados do projeto",
    label: "LEGITIMO"
  },

  {
    text: "Nosso encontro esta confirmado para sexta feira",
    label: "LEGITIMO"
  },

  {
    text: "A nota fiscal foi enviada para seu email",
    label: "LEGITIMO"
  }
];

// ==========================================
// 6. APLICAÇÃO
// ==========================================

class TextClassifierApp {

  private preprocessor: TextPreprocessor;

  private bagOfWords: BagOfWords;

  private knn: KNNClassifier;

  constructor() {

    this.preprocessor = new TextPreprocessor();

    this.bagOfWords = new BagOfWords();

    this.knn = new KNNClassifier(3);

    this.train();

    this.setupInterface();
  }

  private train(): void {

    const tokenizedTexts = trainingData.map(
      example =>
        this.preprocessor.tokenize(example.text)
    );

    this.bagOfWords.fit(tokenizedTexts);

    const vectors = tokenizedTexts.map(
      tokens =>
        this.bagOfWords.transform(tokens)
    );

    const labels = trainingData.map(
      example => example.label
    );

    const texts = trainingData.map(
      example => example.text
    );

    this.knn.fit(
      vectors,
      labels,
      texts
    );
  }

  private setupInterface(): void {

    const button = document.getElementById(
      "classifyButton"
    ) as HTMLButtonElement;

    button.addEventListener(
      "click",
      () => this.classifyText()
    );
  }

  private classifyText(): void {

    const input = document.getElementById(
      "messageInput"
    ) as HTMLTextAreaElement;

    const resultElement = document.getElementById(
      "result"
    ) as HTMLDivElement;

    const text = input.value.trim();

    if (!text) {

      resultElement.textContent =
        "Digite uma mensagem para classificar.";

      return;
    }

    const tokens = this.preprocessor.tokenize(text);

    const vector = this.bagOfWords.transform(tokens);

    const result = this.knn.predict(vector);

    const percentage = Math.round(
      result.confidence * 100
    );

    const resultClass =
      result.label === "GOLPE"
        ? "scam"
        : "legitimate";

    resultElement.className = `result ${resultClass}`;

    resultElement.innerHTML = `
      <h2>Resultado: ${result.label}</h2>

      <p>
        Votos dos vizinhos:
        ${percentage}% para a categoria escolhida.
      </p>

      <h3>Palavras processadas</h3>

      <p>${tokens.join(", ") || "Nenhuma palavra reconhecida"}</p>

      <h3>Vizinhos mais próximos</h3>

      <ul>
        ${result.neighbors.map(neighbor => `
          <li>
            ${neighbor.label} -
            ${neighbor.text}
            (distância: ${neighbor.distance.toFixed(3)})
          </li>
        `).join("")}
      </ul>
    `;
  }
}

// Inicializa a aplicação.

new TextClassifierApp();
