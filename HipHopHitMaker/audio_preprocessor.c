#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sndfile.h>

typedef struct {
    float*  samples;
    int     num_samples;
    int     sample_rate;
    int     channels;
    int     error_code;
    char    error_msg[256];
} AudioBuffer;

static int  is_ffmpeg_format(const char* filepath);
static AudioBuffer* read_with_sndfile(const char* filepath);
static AudioBuffer* read_with_ffmpeg(const char* filepath);

static AudioBuffer* make_error(int code, const char* msg) {
    AudioBuffer* buf = (AudioBuffer*)calloc(1, sizeof(AudioBuffer));
    buf->error_code = code;
    strncpy(buf->error_msg, msg, 255);
    return buf;
}

AudioBuffer* load_audio_file(const char* filepath) {
    if (!filepath || strlen(filepath) == 0) {
        return make_error(1, "Percorso file vuoto");
    }

    if (is_ffmpeg_format(filepath)) {
        return read_with_ffmpeg(filepath);
    } else {
        return read_with_sndfile(filepath);
    }
}

void free_audio_buffer(AudioBuffer* buf) {
    if (buf) {
        if (buf->samples) free(buf->samples);
        free(buf);
    }
}

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

static int is_ffmpeg_format(const char* filepath) {
    const char* ext = strrchr(filepath, '.');
    if (!ext) return 1;
    ext++;

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

static AudioBuffer* read_with_sndfile(const char* filepath) {
    SF_INFO info;
    memset(&info, 0, sizeof(SF_INFO));

    SNDFILE* sf = sf_open(filepath, SFM_READ, &info);
    if (!sf) {
        char msg[256];
        snprintf(msg, sizeof(msg), "libsndfile: %.240s", sf_strerror(NULL));
        return make_error(2, msg);
    }

    long total_samples = (long)info.frames * info.channels;
    float* buffer = (float*)malloc(total_samples * sizeof(float));
    if (!buffer) {
        sf_close(sf);
        return make_error(2, "malloc fallito");
    }

    sf_count_t frames_read = sf_readf_float(sf, buffer, info.frames);
    sf_close(sf);

    if (frames_read <= 0) {
        free(buffer);
        return make_error(2, "Nessun frame letto da libsndfile");
    }

    AudioBuffer* result = (AudioBuffer*)calloc(1, sizeof(AudioBuffer));
    result->samples     = buffer;
    result->num_samples = (int)(frames_read * info.channels);
    result->sample_rate = info.samplerate;
    result->channels    = info.channels;
    result->error_code  = 0;

    return result;
}

static AudioBuffer* read_with_ffmpeg(const char* filepath) {
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
             "ffmpeg -i \"%s\" -f f32le -ar 44100 -ac 1 -loglevel quiet pipe:1",
             filepath);

    FILE* pipe = popen(cmd, "r");
    if (!pipe) {
        return make_error(2, "Impossibile avviare ffmpeg. Installalo con: apt install ffmpeg");
    }

    const int CHUNK = 4096;
    float* buffer   = NULL;
    int    total    = 0;
    int    capacity = 0;

    float temp[4096];
    size_t n;

    while ((n = fread(temp, sizeof(float), CHUNK, pipe)) > 0) {
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
    result->sample_rate = 44100;
    result->channels    = 1;
    result->error_code  = 0;

    return result;
}
