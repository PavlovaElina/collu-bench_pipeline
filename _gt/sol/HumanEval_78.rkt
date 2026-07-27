#lang racket
(define (hex_key num)
  (if (string? num)
      (for/sum ([c (in-string num)]
                #:when (member c '(#\2 #\3 #\5 #\7 #\B #\D))) 1)
      0))
