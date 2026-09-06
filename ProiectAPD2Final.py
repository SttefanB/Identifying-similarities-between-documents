import time
import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, MinHashLSH
 
def create_spark_session(app_name, master, shuffle_partitions, driver_host=None):
    builder = (SparkSession.builder # Se initializeaza constructorul sesiunii
    .appName(app_name) # Aplicatia va primi un nume cu care sa fie recunoscuta in interfata grafica
    .master(master) # Acesta reprezinta locul in care va rula 
    .config("spark.sql.shuffle.partitions", shuffle_partitions)) 
    # Se seteaza in cate bucati mai mici sa fie taiate datele atunci cand le va amesteca si combina
 
    if driver_host:
        builder = builder.config("spark.driver.host", driver_host)
        # Daca codul este rulat pe mai multe calculatoare si trimit acest driver_host, ii spun laptop ului 
        # de pe care lucrez sa anunte IP urile din retea unde trebuie sa trimita rezultatele
        
    spark = builder.getOrCreate() # Se creeaza o sesiune Spark
 
    spark.sparkContext.setLogLevel("WARN")
    # Ascundem mesajele inutile din consola 
    
    return spark
    # Returnam sesiunea configurata
    
def generate_mock_data(spark, num_records):
    from pyspark.sql.functions import expr
 
    print(f"[*] Generare {num_records} documente de test...")
 
    # Definim un vocabular comun de cuvinte. Fiecare document va fi un subset ALEATOR din aceste cuvinte,
    # nu o copie identica a unei propozitii. Astfel documentele care impart multe cuvinte vor semana mult,
    # iar cele cu cuvinte diferite vor semana putin -> scorurile Jaccard se intind pe tot intervalul 0..1
    vocab = [
        "spark", "framework", "procesare", "paralela", "distribuita", "bigdata", "cluster", "noduri",
        "python", "pyspark", "analiza", "seturi", "date", "biblioteca", "cod", "script",
        "lsh", "jaccard", "similaritate", "documente", "hash", "minhash", "distanta", "semnatura",
        "algoritm", "executie", "timp", "sarcina", "partitionare", "performanta", "optimizare", "scalabilitate",
        "memorie", "shuffle", "join", "vectorizare", "tokenizare", "frecventa", "model", "antrenare"
    ]
    vocab_sql = "array(" + ", ".join(f"'{w}'" for w in vocab) + ")"
    # Transformam lista de cuvinte intr-un array pe care Spark il poate folosi direct in interogari
 
    df = spark.range(num_records)
    # spark.range genereaza ID-urile direct distribuit pe executori (nu pe driver), deci scaleaza pentru orice numar de documente
 
    mock_df = df.withColumn(
        "text",
        expr(f"array_join(slice(shuffle({vocab_sql}), 1, cast(floor(rand()*10)+5 as int)), ' ')")
    )
    # Pentru fiecare rand: shuffle amesteca aleator vocabularul, slice ia primele k cuvinte (k intre 5 si 14),
    # iar array_join le lipeste intr-un singur text. Fiecare document primeste astfel un alt subset de cuvinte.
 
    return mock_df.select("id", "text") # Se va returna tabelul de test 
 
def process_and_find_similarities(spark, df, similarity_threshold, top_n): 
   # Functia descopera similaritatile la nivelul fisierelor de tip text
    start_time_total = time.time()
 
 
    print("[*] Etapa 1: Tokenizare și eliminare Stop Words...")
    tokenizer = RegexTokenizer(inputCol="text", outputCol="words", pattern="\\W")
    # Sparge propozitia in cuvinte separate
    words_df = tokenizer.transform(df)
    
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    # Elimina cuvintele de legatura cum sunt "si", "de" etc
    filtered_df = remover.transform(words_df)
 
    print("[*] Etapa 2: Vectorizare HashingTF...")
 
    hashingTF = HashingTF(inputCol="filtered_words", outputCol="features", numFeatures=10000) 
    features_df = hashingTF.transform(filtered_df)
    # Transforma cuvintele ramase in vectori de numere si fiecare cuvant primeste un cod unic intr un tabel
 
    features_df.cache() 
    # Pune toate aceste linii direct in memoria RAM pentru a nu pierde timpul sa le citeasca iar
    print(f"    Documente procesate: {features_df.count()}")
 
    print("[*] Etapa 3: Antrenare model LSH...")
    lsh_start = time.time()
 
    mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=5) 
    model = mh.fit(features_df) 
    # Creeaza algoritmul LSH. Pune textele in locuri diferite, dar in cazul in care seamana le pune in acelasi loc
    # Algoritmul invata pe baza datelor si creeaza acele locuri
    
    lsh_time = time.time() - lsh_start
    print(f"    Timp antrenare LSH: {lsh_time:.2f} secunde")
 
    print(f"[*] Etapa 4: Căutare perechi cu similaritate >= {similarity_threshold}...")
    join_start = time.time()
    
    jaccard_distance_threshold = 1.0 - similarity_threshold
    # Distanta maxima acceptata
    
    similar_pairs = model.approxSimilarityJoin( 
        features_df, features_df, jaccard_distance_threshold, distCol="JaccardDistance" 
    )
    # Spark ia textele din care se afla in acelasi loc si le compara, calculand cat de mult seamana
    # cu ajutorul distantei Jaccard
 
    similar_pairs = similar_pairs.filter(col("datasetA.id") < col("datasetB.id")) 
    # Se elimina dublurile pentru ca programul sa nu mi zica de doua ori ca seamana propozitiile intre ele
    
    total_pairs = similar_pairs.count()
    join_time = time.time() - join_start
    print(f"    Timp calcul similarități (Join & Shuffle): {join_time:.2f} secunde")
    print(f"    Număr total perechi similare găsite: {total_pairs}")
 
    print(f"[*] Afisare Top {top_n} cele mai similare documente:")
    similar_pairs.orderBy("JaccardDistance").select(
                     col("datasetA.id").alias("id_Doc_1"),
                     col("datasetB.id").alias("id_Doc_2"),
                     (1 - col("JaccardDistance")).alias("Similarity_Score")
                 ).show(top_n)
 
    # Sorteaza rezultatele incepand cu cele mai asemanatoare si le aranjeaza pe 3 coloane 
 
    total_time = time.time() - start_time_total
    print("-" * 50)
    print(f"TIMP TOTAL EXECUTIE: {total_time:.2f} secunde")
    print("-" * 50)
    # Se opreste cronometrul pornit la inceputul functiei si se afiseaza in consola
    
    return total_time
 
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Document Similarity with PySpark LSH")
    parser.add_argument("--nodes", type=int, default=2, help="Numarul de fire de executie/noduri")
    parser.add_argument("--partitions", type=int, default=8, help="Numarul de partitii de shuffle")
    parser.add_argument("--records", type=int, default=10000, help="Numarul de documente de test (Chunk Size)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Pragul de similaritate (0.0 la 1.0)")
    parser.add_argument("--master", type=str, default=None, help="URL master Spark (ex: spark://192.168.1.10:7077). Implicit: local[nodes]") 
    parser.add_argument("--driver-host", type=str, default=None, help="IP-ul masinii driver in retea (mod distribuit)") 
    args = parser.parse_args()
    # Ia valorile tastate in consola si le salveaza intr-o lista 
    master_url = args.master if args.master else f"local[{args.nodes}]"
    # Se stabileste unde se va rula programul 
    print(f"=== INIȚIALIZARE RUN ===")
    print(f"Master: {master_url}")
    print(f"Noduri/Cores: {args.nodes}")
    print(f"Partiții Shuffle: {args.partitions}")
    print(f"Dimensiune Date: {args.records} documente")
    print("=========================\n")
 
    spark_session = create_spark_session("DocSimilarityLSH", master_url, args.partitions, args.driver_host)
    # Se apeleaza functia creeata la inceput 
    try:
        data_df = generate_mock_data(spark_session, args.records)
        # Se genereaza un numar de documente
        total_time = process_and_find_similarities(spark_session, data_df, args.threshold, top_n=10)
        # Se apeleaza functia care calculeaza timpul pentru algoritmul LSH
        import csv, os
        exists = os.path.exists("results.csv")
        with open("results.csv", "a", newline="") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(["nodes", "partitions", "records", "total_time"])
            w.writerow([args.nodes, args.partitions, args.records, round(total_time, 2)])
        # Se salveaza raportul intr un fisier CSV
    finally:
        spark_session.stop()
 