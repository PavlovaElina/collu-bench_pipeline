#lang racket
(define (words_in_sentence sentence)
  (define (prime? x) (and (> x 1) (for/and ([d (in-range 2 (+ 1 (integer-sqrt x)))]) (not (= 0 (modulo x d))))))
  (string-join (filter (lambda (w) (prime? (string-length w))) (string-split sentence)) " "))
