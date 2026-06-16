#lang racket
(define (words_string s)
  (filter (lambda (w) (> (string-length w) 0))
          (regexp-split #px"[ ,]+" (string-trim s))))
