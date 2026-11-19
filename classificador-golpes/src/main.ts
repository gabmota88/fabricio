
import "./style.css";

// ==========================================
// 1. TIPOS DO SISTEMA
// ==========================================

// Define as duas possíveis categorias usadas pelo resultado da classificação.
type Label = "GOLPE" | "LEGITIMO";

// Representa um exemplo utilizado para ensinar o classificador.
interface TrainingExample {
  text: string;
  label: Label;
}

// Representa uma mensagem de treinamento comparada com a mensagem enviada.
interface Neighbor {
  text: string;
  label: Label;
  distance: number;
}

// Agrupa todas as informações devolvidas após uma classificação.
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

  // Palavras comuns que não ajudam a diferenciar golpe de mensagem legítima.
  private stopWords: Set<string> = new Set([
    "a", "o", "as", "os",
    "de", "do", "da", "dos", "das",
    "e", "em", "no", "na", "nos", "nas",
    "um", "uma", "para", "por",
    "que", "com", "se", "sua", "seu",
    "meu", "minha", "ao", "aos",
    "foi", "ser", "estar"
  ]);

  // Corrige erros comuns que não devem alterar o significado da mensagem.
  private spellingCorrections: Record<string, string> = {
    "mellhorar": "melhorar"
  };

  public normalize(text: string): string {

    // Converte a mensagem em um formato uniforme antes da análise.
    return text
      // Coloca todo o texto em letras minúsculas.
      .toLowerCase()
      // Separa letras acentuadas de seus sinais, como o "ç" em "c" e "̧".
      .normalize("NFD")
      // Remove os sinais de acentuação separados na etapa anterior.
      .replace(/[\u0300-\u036f]/g, "")
      // Substitui pontuação e outros caracteres especiais por espaços.
      .replace(/[^a-z0-9\s]/g, " ")
      // Une espaços múltiplos em um único espaço.
      .replace(/\s+/g, " ")
      // Remove espaços extras no início e no fim.
      .trim();
  }

  public tokenize(text: string): string[] {

    // Normaliza a mensagem antes de separá-la em palavras.
    const normalized = this.normalize(text);

    // Divide o texto e remove palavras que não contribuem para a classificação.
    return normalized
      .split(" ")
      .filter(word => word.length > 1)
      .filter(word => !this.stopWords.has(word))
      // Aplica correções simples antes da stemização.
      .map(word => this.spellingCorrections[word] ?? word)
      // Reduz cada palavra restante ao seu radical simplificado.
      .map(word => this.stem(word));
  }

  // Executa uma stemização simplificada removendo terminações comuns em português.
  private stem(word: string): string {

    // Lista terminações específicas, começando pelas mais longas.
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

    // Testa cada terminação até encontrar uma que possa ser removida.
    for (const suffix of suffixes) {

      if (
        word.endsWith(suffix) &&
        word.length - suffix.length >= 3
      ) {
        // Mantém pelo menos três caracteres como raiz da palavra.
        return word.slice(0, -suffix.length);
      }
    }

    return word;
  }
}

// Analisa sinais gerais de golpe a partir do contexto da mensagem.
class ScamSignalAnalyzer {

  private signalGroups: Record<string, string[]> = {
    urgencia: [
      "urgente", "agora", "imediatamente", "rapido", "hoje"
    ],
    acao: [
      "clique", "acesse", "confirme", "transfira",
      "pague", "informe", "envie", "resgate", "atualize"
    ],
    dados_sensiveis: [
      "senha", "codigo", "cpf", "dados bancarios",
      "cartao", "pix", "conta"
    ],
    ameaca: [
      "bloqueado", "bloqueada", "cancelamento",
      "suspensa", "suspenso", "retida", "taxa", "regularize"
    ],
    recompensa: [
      "premio", "sorteio", "ganhou", "selecionado", "ganhei"
    ]
  };

  public analyze(text: string): { score: number; indicators: string[] } {

    const normalized = text
      .toLowerCase()
      .normalize("NFD")
      .replace(/[\u0300-\u036f]/g, "");

    const found: string[] = [];
    let score = 0;

    Object.entries(this.signalGroups).forEach(([group, phrases]) => {

      const detected = phrases.some(phrase =>
        normalized.includes(phrase)
      );

      if (detected) {
        found.push(group);
        score++;
      }
    });

    return {
      score,
      indicators: found
    };
  }
}

// ==========================================
// 3. BAG OF WORDS
// ==========================================

class BagOfWords {

  // Guarda todas as palavras conhecidas, uma vez por posição no vetor.
  private vocabulary: string[] = [];

  // Associa cada palavra conhecida ao índice que ela ocupa no vetor.
  private vocabularyIndex: Map<string, number> = new Map();

  public fit(documents: string[][]): void {

    // Usa um conjunto para coletar palavras diferentes sem repeti-las.
    const words = new Set<string>();

    documents.forEach(document => {

      document.forEach(word => {
        words.add(word);
      });

    });

    // Cria uma lista ordenada de palavras para tornar os índices previsíveis.
    this.vocabulary = Array.from(words).sort();

    // Reinicia o índice caso o modelo seja treinado mais de uma vez.
    this.vocabularyIndex.clear();

    this.vocabulary.forEach((word, index) => {

      // Define em qual posição do vetor cada palavra será contabilizada.
      this.vocabularyIndex.set(word, index);

    });
  }

  public transform(tokens: string[]): number[] {

    // Cria um vetor numérico preenchido inicialmente com zeros.
    const vector = new Array(
      this.vocabulary.length
    ).fill(0);

    tokens.forEach(word => {

      // Descobre a posição da palavra no vocabulário.
      const index = this.vocabularyIndex.get(word);

      // Conta a palavra somente quando ela pertence ao vocabulário aprendido.
      if (index !== undefined) {
        vector[index]++;
      }

    });

    return vector;
  }

  public getVocabulary(): string[] {

    // Devolve uma cópia do vocabulário para evitar alterações externas.
    return [...this.vocabulary];
  }
}

// ==========================================
// 4. CLASSIFICADOR KNN
// ==========================================

class KNNClassifier {

  // Vetores numéricos das mensagens usadas no treinamento.
  private trainingVectors: number[][] = [];

  // Categoria correspondente a cada vetor de treinamento.
  private trainingLabels: Label[] = [];

  // Texto original correspondente a cada vetor de treinamento.
  private trainingTexts: string[] = [];

  // Quantidade de vizinhos considerados na votação.
  private k: number;

  constructor(k: number = 3) {
    this.k = k;
  }

  public fit(
    vectors: number[][],
    labels: Label[],
    texts: string[]
  ): void {

    // Guarda os dados necessários para comparar novas mensagens.
    this.trainingVectors = vectors;
    this.trainingLabels = labels;
    this.trainingTexts = texts;
  }

  private cosineDistance(
    vectorA: number[],
    vectorB: number[]
  ): number {

    // Acumula o produto escalar entre os dois vetores.
    let dotProduct = 0;

    // Acumula a magnitude de cada vetor.
    let normA = 0;

    let normB = 0;

    for (let i = 0; i < vectorA.length; i++) {

      // Soma a multiplicação das posições correspondentes.
      dotProduct += vectorA[i] * vectorB[i];

      // Eleva os valores ao quadrado para calcular a magnitude.
      normA += vectorA[i] ** 2;

      normB += vectorB[i] ** 2;
    }

    // Multiplica as magnitudes usadas no cálculo da similaridade.
    const denominator = Math.sqrt(normA) * Math.sqrt(normB);

    // Retorna distância máxima quando pelo menos um vetor não possui palavras.
    if (denominator === 0) {
      return 1;
    }

    // Converte a similaridade de cosseno em uma distância.
    const similarity = dotProduct / denominator;

    return 1 - similarity;
  }

  public predict(
    inputVector: number[]
  ): ClassificationResult {

    // Sem palavras conhecidas, não existe evidência para escolher GOLPE.
    // Isso evita empates artificiais entre os primeiros exemplos da base.
    if (inputVector.every(value => value === 0)) {
      return {
        label: "LEGITIMO",
        confidence: 0,
        neighbors: [],
        tokens: []
      };
    }

    // Monta a lista de vizinhos encontrados no conjunto de treinamento.
    const neighbors: Neighbor[] = [];

    this.trainingVectors.forEach(
      (trainingVector, index) => {

        // Mede a distância entre a mensagem atual e cada exemplo conhecido.
        const distance = this.cosineDistance(
          inputVector,
          trainingVector
        );

        // Guarda o texto, a categoria e a distância de cada vizinho.
        neighbors.push({
          text: this.trainingTexts[index],
          label: this.trainingLabels[index],
          distance
        });

      }
    );

    // Ordena os vizinhos do mais próximo para o mais distante.
    neighbors.sort(
      (a, b) => a.distance - b.distance
    );

    // Seleciona até a quantidade definida pela variável k.
    const nearestNeighbors = neighbors.slice(
      0,
      Math.min(this.k, neighbors.length)
    );

    // Inicializa a contagem de votos de cada categoria.
    const votes: Record<Label, number> = {
      GOLPE: 0,
      LEGITIMO: 0
    };

    // Soma um voto para a categoria de cada vizinho selecionado.
    nearestNeighbors.forEach(neighbor => {

      votes[neighbor.label]++;

    });

    // Escolhe GOLPE somente quando houver mais votos dessa categoria.
    const label: Label =
      votes.GOLPE > votes.LEGITIMO
        ? "GOLPE"
        : "LEGITIMO";

    // Calcula a porcentagem de votos da categoria vencedora.
    const confidence =
      nearestNeighbors.length > 0
        ? votes[label] / nearestNeighbors.length
        : 0;

    // Devolve a categoria, a confiança e os vizinhos usados na decisão.
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

// Armazena frases conhecidas e suas respectivas classificações.
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
  },

  {
    text: "Estude para melhorar seus conhecimentos",
    label: "LEGITIMO"
  },

  {
    text: "Estudar todos os dias ajuda a melhorar",
    label: "LEGITIMO"
  }
];

// ==========================================
// 6. APLICAÇÃO
// ==========================================

class TextClassifierApp {

  // Responsável por normalizar e separar o texto em palavras.
  private preprocessor: TextPreprocessor;

  // Avalia sinais gerais que podem indicar uma tentativa de golpe.
  private scamSignalAnalyzer: ScamSignalAnalyzer;

  // Responsável por criar os vetores numéricos das mensagens.
  private bagOfWords: BagOfWords;

  // Responsável por comparar mensagens e votar na categoria.
  private knn: KNNClassifier;

  constructor() {

    // Cria as três partes usadas no fluxo de classificação.
    this.preprocessor = new TextPreprocessor();
    this.scamSignalAnalyzer = new ScamSignalAnalyzer();
    this.bagOfWords = new BagOfWords();
    this.knn = new KNNClassifier(3);

    // Treina o modelo antes de iniciar a interação da página.
    this.train();

    // Conecta o processamento às informações exibidas na interface.
    this.setupInterface();
  }

  private train(): void {

    // Transformar cada texto de treinamento em uma lista de palavras processadas.
    const tokenizedTexts = trainingData.map(
      example =>
        this.preprocessor.tokenize(example.text)
    );

    // Construir o vocabulário a partir de todas as palavras de treinamento.
    this.bagOfWords.fit(tokenizedTexts);

    // Converter os textos processados em vetores Bag of Words.
    const vectors = tokenizedTexts.map(
      tokens =>
        this.bagOfWords.transform(tokens)
    );

    // Separar as categorias correspondentes aos exemplos.
    const labels = trainingData.map(
      example => example.label
    );

    // Separar os textos originais para exibí-los entre os vizinhos próximos.
    const texts = trainingData.map(
      example => example.text
    );

    // Registrar vetores, categorias e textos no classificador KNN.
    this.knn.fit(
      vectors,
      labels,
      texts
    );
  }

  private setupInterface(): void {

    // Obtém o botão utilizado pelo usuário para iniciar a classificação.
    const button = document.getElementById(
      "classifyButton"
    ) as HTMLButtonElement;

    // Associa o clique do botão à execução da classificação.
    button.addEventListener(
      "click",
      () => this.classifyText()
    );
  }

  private classifyText(): void {

    // Obtém o campo onde o usuário digitou a mensagem.
    const input = document.getElementById(
      "messageInput"
    ) as HTMLTextAreaElement;

    // Obtém o elemento responsável por mostrar o resultado na página.
    const resultElement = document.getElementById(
      "result"
    ) as HTMLDivElement;

    // Remove espaços vazios no início e no fim da mensagem.
    const text = input.value.trim();

    // Interrompe a classificação quando o usuário não digitou nada.
    if (!text) {

      resultElement.textContent =
        "Digite uma mensagem para classificar.";

      return;
    }

    // Converte a mensagem digitada em palavras processadas.
    const tokens = this.preprocessor.tokenize(text);

    // Representa a mensagem como um vetor com contagens das palavras.
    const vector = this.bagOfWords.transform(tokens);

    // Executa a classificação pelo algoritmo KNN.
    const knnResult = this.knn.predict(vector);
    const risk = this.scamSignalAnalyzer.analyze(text);

    // Combina o KNN com sinais contextuais em vez de depender de uma frase exata.
    const result: ClassificationResult =
      risk.score >= 2
        ? {
            ...knnResult,
            label: "GOLPE",
            confidence: Math.max(
              knnResult.confidence,
              Math.min(1, risk.score / 5)
            )
          }
        : risk.score === 0
          ? {
              ...knnResult,
              label: "LEGITIMO",
              confidence: Math.max(0.67, knnResult.confidence)
            }
          : knnResult;

    // Converte a confiança da votação em uma porcentagem inteira.
    const percentage = Math.round(
      result.confidence * 100
    );

    // Define uma classe visual conforme o resultado da classificação.
    const resultClass =
      result.label === "GOLPE"
        ? "scam"
        : "legitimate";

    // Aplica a classe que determina as cores do painel de resultado.
    resultElement.className = `result ${resultClass}`;

    // Monta os elementos exibidos ao usuário com o resultado e os detalhes.
    resultElement.innerHTML = `
      <h2>Resultado: ${result.label}</h2>

      <p>
        Votos dos vizinhos:
        ${percentage}% para a categoria escolhida.
      </p>

      <h3>Palavras processadas</h3>

      <p>${tokens.join(", ") || "Nenhuma palavra reconhecida"}</p>

      <h3>Sinais identificados</h3>

      <p>${risk.indicators.join(", ") || "Nenhum sinal de golpe identificado"}</p>

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

// Cria a aplicação para que o treinamento e os eventos da interface sejam iniciados.
new TextClassifierApp();
