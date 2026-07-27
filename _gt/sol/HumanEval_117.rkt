#lang racket
(define (select_words s n)
  (define (consonants w)
    (for/sum ([c (in-string w)]
              #:when (and (char-alphabetic? c)
                          (not (member (char-downcase c) '(#\a #\e #\i #\o #\u))))) 1))
  (filter (lambda (w) (= (consonants w) n)) (string-split s)))
