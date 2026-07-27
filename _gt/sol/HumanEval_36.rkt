#lang racket
(define (fizz_buzz n)
  (for/sum ([i (in-range n)]
            #:when (or (= 0 (modulo i 11)) (= 0 (modulo i 13))))
    (for/sum ([c (in-string (number->string i))] #:when (char=? c #\7)) 1)))
