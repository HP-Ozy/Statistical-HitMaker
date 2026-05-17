/*
 * audio_preprocessor.c
 * ─────────────────────────────────────────────────────────────────────────────
 * Libreria condivisa (.so) per leggere file audio in vari formati
 * e restituire campioni PCM float32 grezzi a Python via ctypes.
 *
 * TEORIA APPLICATA:
 *  - PCM (Pulse Code Modulation): rappresentazione digitale del segnale audio
 *    come sequenza di valori numerici campionati nel tempo.
 *  - Campionamento: ogni "frame" è il valore della pressione sonora
 *    in un istante preciso. A 44100 Hz → 44100 campioni al secondo.
 *  - I campioni float32 sono normalizzati in [-1.0, +1.0].
 *
 * FORMATI SUPPORTATI:
 *  - libsndfile  → WAV, FLAC, AIFF, OGG, AU, W64, RF64
 *  - ffmpeg pipe → MP3, AAC, M4A, WMA, OPUS, MP4 (qualsiasi cosa ffmpeg capisca)
 *
 * COMPILAZIONE:
 *  gcc -O2 -shared -fPIC -o libaudio.so audio_preprocessor.c -lsndfile
 *
 * ─────────────────────────────────────────────────────────────────────────────
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sndfile.h>   /* libsndfile: astrazione su WAV/FLAC/OGG/AIFF */

/* ── Struttura restituita a Python ───────────────────────────────────────── */
typedef struct {
    float*  samples;      /* buffer PCM float32, allocato con malloc()        */
    int     num_samples;  /* numero totale di campioni (frames * channels)    */
    int     sample_rate;  /* frequenza di campionamento originale (Hz)        */
    int     channels;     /* numero di canali (1=mono, 2=stereo, ...)         */
    int     error_code;   /* 0 = successo | 1 = file non trovato | 2 = errore decode */
    char    error_msg[256];
} AudioBuffer;

/* ── Prototipi interni ───────────────────────────────────────────────────── */
static int  is_ffmpeg_format(const char* filepath);
static AudioBuffer* read_with_sndfile(const char* filepath);
static AudioBuffer* read_with_ffmpeg(const char* filepath);

/* ── Costruttore buffer errore ───────────────────────────────────────────── */
static AudioBuffer* make_error(int code, const char* msg) {
    AudioBuffer* buf = (AudioBuffer*)calloc(1, sizeof(AudioBuffer));
    buf->error_code = code;
    strncpy(buf->error_msg, msg, 255);
    return buf;
}

/* ─────────────────────────────────────────────────────────────────────────
 * FUNZIONE PRINCIPALE (chiamata da Python via ctypes)
 *
 * Ritorna un puntatore ad AudioBuffer allocato su heap.
 * Python chiama free_audio_buffer() quando ha finito.
 * ───────────────────────────────────────────────────────────────────────── */
AudioBuffer* load_audio_file(const char* filepath) {
    if (!filepath || strlen(filepath) == 0) {
        return make_error(1, "Percorso file vuoto");
    }

    /* Scegli il decoder in base al formato */
    if (is_ffmpeg_format(filepath)) {
        return read_with_ffmpeg(filepath);
    } else {
        return read_with_sndfile(filepath);
    }
}

/* ── Libera la memoria allocata (chiamato da Python) ─────────────────────── */
void free_audio_buffer(AudioBuffer* buf) {
    if (buf) {
        if (buf->samples) free(buf->samples);
        free(buf);
    }
}

/* ── Ritorna info sul formato senza caricare i campioni ──────────────────── */
void get_format_info(const char* filepath,
                     int* out_samplerate,
                     int* out_channels,
                     int* out_frames,
                     char* out_format_str,
                     int   format_str_len)
{
    SF_INFO info;
    memset(&info, 0, sizeof(SF_INFO));

    SNDFILE* sf = sf_open(filepath, SFM_READ, &info);
    if (!sf) {
        snprintf(out_format_str, format_str_len, "ffmpeg-required");
        *out_samplerate = 0;
        *out_channels   = 0;
        *out_frames     = 0;
        return;
    }

    *out_samplerate = info.samplerate;
    *out_channels   = info.channels;
    *out_frames     = (int)info.frames;

    /* Decodifica il formato in stringa leggibile */
    switch (info.format & SF_FORMAT_TYPEMASK) {
        case SF_FORMAT_WAV:  snprintf(out_format_str, format_str_len, "WAV");  break;
        case SF_FORMAT_FLAC: snprintf(out_format_str, format_str_len, "FLAC"); break;
        case SF_FORMAT_OGG:  snprintf(out_format_str, format_str_len, "OGG");  break;
        case SF_FORMAT_AIFF: snprintf(out_format_str, format_str_len, "AIFF"); break;
        case SF_FORMAT_AU:   snprintf(out_format_str, format_str_len, "AU");   break;
        default:             snprintf(out_format_str, format_str_len, "OTHER"); break;
    }

    sf_close(sf);
}

/* ─────────────────────────────────────────────────────────────────────────
 * IMPLEMENTAZIONI INTERNE
 * ───────────────────────────────────────────────────────────────────────── */

/*
 * Controlla se il file è un formato che libsndfile NON supporta
 * e che quindi richiede ffmpeg come decoder esterno.
 */
static int is_ffmpeg_format(const char* filepath) {
    /* Cerca l'estensione (ultima occorrenza di '.') */
    const char* ext = strrchr(filepath, '.');
    if (!ext) return 1;  /* nessuna estensione → proviamo ffmpeg */
    ext++;                /* salta il punto */

    /* Lista di estensioni che richiedono ffmpeg */
    const char* ffmpeg_exts[] = {
        "mp3", "MP3",
        "aac", "AAC",
        "m4a", "M4A",
        "wma", "WMA",
        "opus","OPUS",
        "mp4", "MP4",
        "webm","WEBM",
        NULL
    };

    for (int i = 0; ffmpeg_exts[i] != NULL; i++) {
        if (strcmp(ext, ffmpeg_exts[i]) == 0) return 1;
    }
    return 0;
}

/*
 * Legge il file con libsndfile.
 *
 * TEORIA: SF_READ_FLOAT normalizza automaticamente tutti i formati
 * (int16, int24, int32, float64) in float32 [-1.0, +1.0].
 * Questo è il formato atteso dai modelli ML.
 */
static AudioBuffer* read_with_sndfile(const char* filepath) {
    SF_INFO info;
    memset(&info, 0, sizeof(SF_INFO));

    /* Apri il file — libsndfile rileva automaticamente il formato */
    SNDFILE* sf = sf_open(filepath, SFM_READ, &info);
    if (!sf) {
        char msg[256];
        snprintf(msg, sizeof(msg), "libsndfile: %.240s", sf_strerror(NULL));
        return make_error(2, msg);
    }

    /* Alloca il buffer: frames × channels campioni float32 */
    long total_samples = (long)info.frames * info.channels;
    float* buffer = (float*)malloc(total_samples * sizeof(float));
    if (!buffer) {
        sf_close(sf);
        return make_error(2, "malloc fallito");
    }

    /*
     * Leggi TUTTI i campioni in una sola chiamata.
     * sf_readf_float() legge per "frame" (un frame = tutti i canali in un istante).
     * Ritorna il numero di frame letti.
     */
    sf_count_t frames_read = sf_readf_float(sf, buffer, info.frames);
    sf_close(sf);

    if (frames_read <= 0) {
        free(buffer);
        return make_error(2, "Nessun frame letto da libsndfile");
    }

    /* Popola la struttura risultato */
    AudioBuffer* result = (AudioBuffer*)calloc(1, sizeof(AudioBuffer));
    result->samples     = buffer;
    result->num_samples = (int)(frames_read * info.channels);
    result->sample_rate = info.samplerate;
    result->channels    = info.channels;
    result->error_code  = 0;

    return result;
}

/*
 * Legge il file tramite ffmpeg, che lo decodifica e lo passa
 * attraverso una pipe in formato WAV raw (PCM s16le).
 *
 * TEORIA: "pipe" = meccanismo IPC (Inter-Process Communication) del kernel.
 * popen() crea un processo figlio (ffmpeg) e ci collega stdout via file descriptor.
 * Leggiamo i byte raw di PCM direttamente dallo stdout di ffmpeg.
 *
 * Formato di uscita scelto: f32le = float 32-bit little-endian
 * → già nel formato che vogliamo, nessuna conversione necessaria.
 */
static AudioBuffer* read_with_ffmpeg(const char* filepath) {
    /*
     * Comando ffmpeg:
     *  -i <file>          → file di ingresso
     *  -f f32le           → formato output: float32 little-endian raw
     *  -ar 44100          → resample a 44100 Hz (Python farà il downsampling)
     *  -ac 1              → mono (media dei canali se stereo)
     *  -loglevel quiet    → nessun output su stderr
     *  pipe:1             → scrivi su stdout (pipe)
     */
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
             "ffmpeg -i \"%s\" -f f32le -ar 44100 -ac 1 -loglevel quiet pipe:1",
             filepath);

    FILE* pipe = popen(cmd, "r");
    if (!pipe) {
        return make_error(2, "Impossibile avviare ffmpeg. Installalo con: apt install ffmpeg");
    }

    /* Leggi i dati dalla pipe in chunk da 4096 float */
    const int CHUNK = 4096;
    float* buffer   = NULL;
    int    total    = 0;
    int    capacity = 0;

    float temp[4096];
    size_t n;

    while ((n = fread(temp, sizeof(float), CHUNK, pipe)) > 0) {
        /* Espandi il buffer se necessario (pattern realloc dinamico) */
        if (total + (int)n > capacity) {
            capacity = (capacity == 0) ? CHUNK * 10 : capacity * 2;
            buffer   = (float*)realloc(buffer, capacity * sizeof(float));
            if (!buffer) {
                pclose(pipe);
                return make_error(2, "realloc fallito durante lettura ffmpeg");
            }
        }
        memcpy(buffer + total, temp, n * sizeof(float));
        total += (int)n;
    }

    pclose(pipe);

    if (total == 0) {
        if (buffer) free(buffer);
        return make_error(2, "ffmpeg non ha prodotto output. File corrotto o formato non supportato.");
    }

    AudioBuffer* result = (AudioBuffer*)calloc(1, sizeof(AudioBuffer));
    result->samples     = buffer;
    result->num_samples = total;
    result->sample_rate = 44100;  /* come specificato nel comando ffmpeg */
    result->channels    = 1;      /* mono, come specificato */
    result->error_code  = 0;

    return result;
}
