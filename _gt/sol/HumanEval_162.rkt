#lang racket
(require file/md5)
(define (string_to_md5 text)
  (if (= (string-length text) 0) #f
      (bytes->string/utf-8 (md5 (string->bytes/utf-8 text)))))
